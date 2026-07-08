import torch
import torch.optim as optim

import os,time,pdb, logging
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
        optimizer = optim.AdamW(model.parameters(), lr = config.lr)
    elif config.optim == "SGD":
        wd = config.weight_decay if config.weight_decay is not None else 5e-4
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


    if config.cyclic :
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, steps_per_epoch=len(train_loader), epochs=config.epochs, pct_start = 0.5)
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
        output_path = os.path.join(str(path.parent.absolute()), 'results', config.dataset, config.config_name.split('.')[0])
        os.makedirs(output_path, exist_ok=True)
        logfile = os.path.join(output_path, 'result_summary.log')
        if os.path.exists(logfile):
            os.remove(logfile)
        logging.basicConfig(
                format='[%(asctime)s] - %(message)s',
                datefmt='%Y/%m/%d %H:%M:%S',
                level=logging.INFO,
                filename=os.path.join(output_path, 'output.log'))

    main(config,logger)