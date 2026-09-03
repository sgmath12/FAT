import torch
import torch.optim as optim

import os,time,pdb, logging
from datetime import datetime
import dataset
import copy
import numpy as np
from pathlib import Path
import methods
from utils import *

# we fix the random seed to 0, this method can keep the results consistent in the same conputer.
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------
# ONE TRAINING PER GPU (2026-09-03).
#
# Four separate times, independent launchers have put two trainings on this card at once: a scratchpad
# waiter racing the master queue, a leftover waiter firing when an earlier driver exited, and so on.
# Each time the fix was a better `while pgrep` in whichever driver was at fault, and each time the next
# launcher did not know about it.  Guards in drivers cannot work, because the failure is always a
# launcher nobody remembered.
#
# So the guard lives here instead, in the one place every run passes through, and it is a real mutex:
# an exclusive flock the process holds for its lifetime.  The kernel releases it when the process
# exits, crashes or is killed, so there is no stale-lock failure mode -- which is why this is flock and
# not the PID file that was tried on 09-02 and immediately went stale.
#
# Behaviour is to WAIT, not to fail.  A queued run blocking until the card is free is exactly what
# every one of those waiter scripts was trying to express, so with this in place they become correct
# no matter how many of them exist or who wrote them.
#
# `FAT_ALLOW_CONCURRENT=1` skips it, for the deliberate two-job case.  Evaluation-only and diagnostic
# scripts do not import this path and are unaffected.
# The lock is PER DEVICE, keyed on CUDA_VISIBLE_DEVICES.  A single global lock would serialize a
# multi-GPU machine down to one job, which is the opposite of the point; four trainings on four cards
# is correct and four on one card is not.  Launch as
#     CUDA_VISIBLE_DEVICES=2 python main.py --config_name ... &
# and each device gets its own lock file.  Unset means device 0.
_gpu_lock_fh = None


def _gpu_lock_path():
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip() or "0"
    return "/tmp/fat_gpu_%s.lock" % dev.replace(",", "_")


def acquire_gpu_lock(poll_seconds=60):
    """Block until this process is the only main.py training on the GPU."""
    global _gpu_lock_fh
    if os.environ.get("FAT_ALLOW_CONCURRENT") == "1":
        print("[gpu-lock] FAT_ALLOW_CONCURRENT=1, 락 건너뜀", flush=True)
        return
    import fcntl
    path = _gpu_lock_path()
    _gpu_lock_fh = open(path, "a+")
    waited = 0
    while True:
        try:
            fcntl.flock(_gpu_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if waited == 0:
                _gpu_lock_fh.seek(0)
                print(f"[gpu-lock] {path}: 다른 학습이 사용 중 "
                      f"({_gpu_lock_fh.read().strip()}), 끝날 때까지 대기", flush=True)
            time.sleep(poll_seconds)
            waited += poll_seconds
    # Runs that started BEFORE this guard existed hold no lock, so the flock alone would let us in
    # beside them.  Wait those out too; this check costs nothing once every run holds the lock.
    # Transitional sweep for runs started before this guard existed, which hold no lock.  Restricted to
    # OUR device by reading each candidate's own CUDA_VISIBLE_DEVICES from /proc, so a job on GPU 1
    # does not wait for one on GPU 0.
    import subprocess
    mine = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip() or "0"

    def _dev_of(pid):
        try:
            with open("/proc/%s/environ" % pid, "rb") as fh:
                for kv in fh.read().split(b"\0"):
                    if kv.startswith(b"CUDA_VISIBLE_DEVICES="):
                        return kv.split(b"=", 1)[1].decode().strip() or "0"
        except Exception:
            return None
        return "0"

    while True:
        try:
            out = subprocess.run(["pgrep", "-f", "main.py --config_name"],
                                 capture_output=True, text=True).stdout.split()
        except Exception:
            break
        others = [q for q in out if q.strip() and int(q) != os.getpid() and _dev_of(q) == mine]
        if not others:
            break
        if waited == 0:
            print(f"[gpu-lock] GPU {mine} 에서 락 없이 도는 학습 {others} 대기", flush=True)
        time.sleep(poll_seconds)
        waited += poll_seconds
    _gpu_lock_fh.seek(0)
    _gpu_lock_fh.truncate()
    _gpu_lock_fh.write(f"pid={os.getpid()} started={datetime.now():%m-%d %H:%M}\n")
    _gpu_lock_fh.flush()
    if waited:
        print(f"[gpu-lock] {path} 획득 (대기 {waited//60}분)", flush=True)
    else:
        print(f"[gpu-lock] {path} 획득", flush=True)


def main(config,npt):
    seed = getattr(config, "seed", 0)          # re-seed here (module-level line 13 runs before config exists)
    import random as _random
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); _random.seed(seed)
    # val=config.val must be forwarded: bilevel methods draw their meta batch from the val split and
    # assume it is HELD OUT. Before this fix (2026-07-06) the main train_loader silently ignored
    # config.val and trained on the full 50000 -- so every bilevel run's "held-out" meta batch
    # actually overlapped the training set (train 391 batches instead of 352 was the tell).
    train_loader, val_loader, test_loader = getattr(dataset,config.dataset)\
        (root= os.path.join(config.data_root, config.dataset), download=True, batch_size = config.batch_size,
         val = bool(getattr(config, "val", False)), config = config)

    path = Path(os.path.realpath(__file__))
    teacher_model, model = get_model(config)

    if config.optim == "AdamW":
        if bool(getattr(config, "nodecay_logscale", False)):
            # Discriminating test (2026-07-13): AdamW's default decoupled wd (0.01) on a
            # LOG-parametrized head scalar (log_g / log_s) pulls the scale toward 1 (= teacher
            # scale) every step -- a different regularizer than wd on a 512-d weight row, and a
            # candidate common cause for all restricted-head -1.8 losses (gainhead/coshead never
            # reproduce the free head's 5x ||w_c|| growth). This flag exempts ONLY those scalars.
            nd = [p for n, p in model.named_parameters() if n.split('.')[-1] in ('log_g', 'log_s')]
            nd_ids = {id(p) for p in nd}
            rest = [p for p in model.parameters() if id(p) not in nd_ids]
            optimizer = optim.AdamW([{'params': rest}, {'params': nd, 'weight_decay': 0.0}], lr = config.lr)
        else:
            optimizer = optim.AdamW(model.parameters(), lr = config.lr)
    elif config.optim == "SGD":
        wd = config.weight_decay if config.weight_decay is not None else 5e-4
        # LBGAT trains its natural branch JOINTLY -- its optimizer holds both parameter sets and its
        # objective carries a CE term on the teacher.  Freezing the teacher would be a different
        # method, and a weaker one, so `joint_teacher` puts the teacher in the optimizer here rather
        # than letting train_lbgat build a second one, which would not share momentum or the schedule.
        if bool(getattr(config, "joint_teacher", False)):
            optimizer = optim.SGD([{'params': model.parameters()},
                                   {'params': teacher_model.parameters()}],
                                  lr=config.lr, momentum=0.9, weight_decay=wd)
            for _p in teacher_model.parameters():
                _p.requires_grad_(True)
        else:
            optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=wd)
    else :
        print ("Not Implemented !")
        return 
    
    best_robust_acc = 0.0
    model_save_path = os.path.join(str(path.parent.absolute()),config.dataset)+ '/checkpoint/' + config.config_name.split('.')[0] + '/'
    os.makedirs(model_save_path, exist_ok=True) 

    d2 = ({
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "alpha": config.alpha,
        "beta" : config.beta,
        "lamda": config.lamda,
        "tau": config.tau,
        "gamma": getattr(config, "gamma", None),
        "eta": getattr(config, "eta", None),
        "temperature": getattr(config, "temperature", None),
        "smooth_k": getattr(config, "smooth_k", None),
        "feat_scale": getattr(config, "feat_scale", None),
        "seed": getattr(config, "seed", None),
        "vuln_swap": getattr(config, "vuln_swap", None),
        "block_norm": getattr(config, "block_norm", None),
        "delta_r": getattr(config, "delta_r", None),
        "delta_meta_lr": getattr(config, "delta_meta_lr", None),
        "tau_meta_lr": getattr(config, "tau_meta_lr", None),
        "bilevel_start": getattr(config, "bilevel_start", None),
        "bilevel_end": getattr(config, "bilevel_end", None),
        "train_eps": getattr(config, "train_eps", None),
        "freeze_lr_epoch": getattr(config, "freeze_lr_epoch", None),
        "wa_start": getattr(config, "wa_start", None),
        "kappa": config.kappa,
        "lr": config.lr,
        "load": config.load,
        "finetune": config.finetune,
        "evaluate": config.evaluate,
        "reformation": config.reformation,
        "student_norm": getattr(config, "student_norm", config.reformation),
        "teacher_norm": getattr(config, "teacher_norm", config.reformation),
        "cyclic": config.cyclic,
        "weight_avg":config.weight_avg,
    })

    logger.info(f"Experiment Configuration: {d2}")

    path = Path(os.path.realpath(__file__))
    path = str(path.parent.absolute())


    clean_acc_teacher = evaluate_clean(teacher_model,test_loader,config)
    print (clean_acc_teacher)
    # import pdb
    # pdb.set_trace()


    # lr_schedule: piecewise (2026-07-26) = ReBAT/RPAT's long-run recipe -- lr_max held flat, then
    # divided by lr_factor at stage1 and by lr_factor^2 at stage2 (their small 1.5 factor, not the
    # usual 10). Added so a 200-epoch scratch run can be schedule-matched to that baseline instead
    # of FAT's OneCycle default. methods.* call scheduler.step() once per BATCH, so this is a
    # per-iteration LambdaLR (t = fractional epoch), which is exactly how ReBAT computes it too.
    # lr_schedule: lbgat (2026-09-03) = LBGAT's own adjust_learning_rate, transcribed exactly from
    # train_lbgat_cifar100.py: 0.1, then 0.02 during epoch 1 only, then 0.01 from 76 and 0.001 from 91.
    # The epoch-1 dip looks like a typo and is theirs; it is kept because it is not ours to fix, and
    # because it is evidently load-bearing.  Under our flat-0.1 protocol LBGAT trains fine on CIFAR-100
    # (8.52 -> 22.66 -> 32.06 clean over the first three evaluations) and dies instantly on CIFAR-10,
    # pinned at 9.9999 from step 0 with an identical config.  Reporting that 10.00 as LBGAT's number
    # would be publishing our optimizer failure as their result.
    if str(getattr(config, "lr_schedule", None) or "") == "lbgat":
        steps_per_epoch = len(train_loader)

        def _lbgat_lr(step):
            t = step / steps_per_epoch
            if 1 <= t < 2:
                return 0.2                      # 0.02 / 0.1
            if t >= 91:
                return 0.01
            if t >= 76:
                return 0.1
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lbgat_lr)
    elif str(getattr(config, "lr_schedule", None) or "") == "piecewise":
        steps_per_epoch = len(train_loader)
        stage1 = int(getattr(config, "stage1", None) or config.epochs // 2)
        stage2 = int(getattr(config, "stage2", None) or config.epochs * 3 // 4)
        lr_factor = float(getattr(config, "lr_factor", None) or 1.5)

        def _piecewise(step):
            t = step / steps_per_epoch
            if t < stage1:
                return 1.0
            elif t < stage2:
                return 1.0 / lr_factor
            return 1.0 / lr_factor ** 2

        logger.info({"lr_schedule": "piecewise", "lr_max": config.lr, "stage1": stage1,
                     "stage2": stage2, "lr_factor": lr_factor})
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _piecewise)
    elif config.cyclic :
        # pct_start defaults to 0.5, our recipe's value.  HAT's own trainer uses OneCycleLR at
        # pct_start=0.25 with lr 0.21, so `pct_start` is exposed rather than hard-wired: forcing our
        # 0.5 onto a baseline whose published recipe says 0.25 is not a fairer comparison, it is a
        # different method.
        _pct = float(getattr(config, "pct_start", 0.0) or 0.0) or 0.5
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, steps_per_epoch=len(train_loader), epochs=config.epochs, pct_start = _pct)
    else :
        scheduler = None


    total_time=0.0
    epoch_time=[]
    log_idx = 0
    
    
    test_model = copy.deepcopy(model)
    exp_avg = model.state_dict()
    # exp_avg = copy.deepcopy(model.state_dict())


    
    # clean_acc, fgsm_acc, pgd_acc, pgd10_acc, pgd50_acc, cw_acc = evaluate(test_model,test_loader,config)
    # d2 = {"last_clean_acc" : clean_acc, "last_fgsm_acc" : fgsm_acc, "last_pgd20_acc":pgd_acc,  "last_pgd10_acc" : pgd10_acc, "last_pgd50_acc" : pgd50_acc, "last_cw_acc":cw_acc}
    # logger.info(d2)

    for epoch in range(config.epochs):

        start = time.time()
        getattr(methods, "train_" + config.method)(model,train_loader,optimizer,teacher_model,epoch,config,scheduler,exp_avg)
        epoch_time.append(time.time() - start)
        total_time += epoch_time[-1]
 
        if epoch % config.interval == 0  :
            if "clean" in config.method:
                test_model.load_state_dict(exp_avg) 
                clean_acc, robust_acc = evaluate_clean(model, test_loader)
                best_model_save_path = model_save_path + '%s_%.4f_best.pkl'%(config.method, clean_acc)
                torch.save(test_model.state_dict(),best_model_save_path)
            else:
                test_model.load_state_dict(exp_avg) 
                clean_acc, robust_acc = evaluate_pgd(test_model,test_loader,config)
                
            if config.log:
                d2 = {"clean_acc":clean_acc, "rob_acc":robust_acc, "task_step" : log_idx}
                log_idx += 1
                logger.info(d2)
   
            if best_robust_acc <= robust_acc :
                best_model_save_path = model_save_path + '%s_best.pkl'%config.method
                torch.save(test_model.state_dict(),best_model_save_path)
                best_robust_acc = robust_acc


    last_model_save_path = model_save_path + '%s_last.pkl'%config.method
    if "clean" in config.method:
        torch.save(model.state_dict(),last_model_save_path)
    else :
        # test_model is only refreshed inside the `epoch % interval` block, so before this fix it
        # held the weights from the LAST INTERVAL EPOCH (e.g. epoch 45 of a 50ep run with
        # interval 5) -- every `_last.pkl` on disk was up to `interval - 1` epochs stale, while the
        # last_*_acc logged below (which loads exp_avg first) described the real final model.
        # Found 2026-08-17: AA re-evaluated from the n2_* checkpoints reported clean 61.75/62.20,
        # matching those runs' epoch-45 interval line exactly instead of their 62.86/62.53 finals.
        # Refresh from exp_avg here so the saved checkpoint IS the model the final eval reports.
        test_model.load_state_dict(exp_avg)
        torch.save(test_model.state_dict(),last_model_save_path)

    epoch_time = np.array(epoch_time)
    total_time = np.array(total_time)
    logger.info({"epoch_time : %.4f hour"%(epoch_time.mean()/3600), "total_time : %.4f hour"%(total_time/3600)})



    if config.evaluate :
        test_model.load_state_dict(exp_avg)
        clean_acc, fgsm_acc, pgd_acc, pgd10_acc, pgd50_acc, cw_acc = evaluate(test_model,test_loader,config)
        d2 = {"last_clean_acc" : clean_acc, "last_fgsm_acc" : fgsm_acc, "last_pgd20_acc":pgd_acc,  "last_pgd10_acc" : pgd10_acc, "last_pgd50_acc" : pgd50_acc, "last_cw_acc":cw_acc}
        logger.info(d2)
        # clean_acc, _ = evaluate_clean(test_model, test_loader, config)
        # d2 = {"last_clean_acc" : clean_acc}
        # logger.info(d2)
        _aa = getattr(config, "aa", None)
        if _aa is None or bool(_aa):   # set `aa: False` in config to skip AA during sweeps
            aa_acc = evaluate_final_aa(test_model, test_loader, config)
            d2 = {"last_aa_acc" : aa_acc}
            logger.info(d2)


    #     test_model.load_state_dict(torch.load(best_model_save_path))

    #     clean_acc, fgsm_acc, pgd_acc, pgd10_acc, pgd50_acc, cw_acc = evaluate(test_model,test_loader,config)
    #     d2 = {"best_clean_acc" : clean_acc, "best_fgsm_acc" : fgsm_acc, "best_pgd20_acc":pgd_acc, "best_pgd10_acc" : pgd10_acc, "best_pgd50_acc" : pgd50_acc, "best_cw_acc":cw_acc}
    #     logger.info(d2)
    #     aa_acc = evaluate_final_aa(test_model, test_loader, config)
    #     d2 = {"best_aa_acc" : aa_acc}
    #     logger.info(d2)


    # logger.info("=" * 10)

if __name__ == "__main__":
    args = load_parser()
    path = Path(os.path.realpath(__file__))
    config = load_config(args)

    if config.log:
        logger = logging.getLogger(__name__)
        arch = str(getattr(config, "arch", "ResNet18") or "ResNet18")
        output_path = os.path.join(str(path.parent.absolute()), 'results', config.dataset, arch, config.config_name.split('.')[0])
        os.makedirs(output_path, exist_ok=True)
        # timestamped filename (not the fixed 'output.log'): basicConfig's default filemode is
        # 'a' (append), so sweep runs sharing one --config_name were silently interleaving into
        # the same file instead of overwriting it -- this gives each run its own log.
        logfile = os.path.join(output_path, datetime.now().strftime('%y%m%d%H%M') + '.log')
        logging.basicConfig(
                format='[%(asctime)s] - %(message)s',
                datefmt='%Y/%m/%d %H:%M:%S',
                level=logging.INFO,
                filename=logfile)

    acquire_gpu_lock()
    main(config,logger)