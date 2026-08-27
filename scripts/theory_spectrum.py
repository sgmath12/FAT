"""Numbers quoted in METHOD.md 260825 (section 8): the reliability-spectrum model.

Reproduces, in order:
  1. Prop. A2 exchange-rate check (closed form vs finite difference)
  2. the soft-threshold frontier table (lambda/eps sweep)
  3. the two-atom (T.1 geometry) degeneracy table
  4. rate universality: the rate is spectrum-independent, the gain is not
  5. Prop. A1 under the adversarial-logistic surrogate (same solution)
  6. Prop. C and the negative result of 8.5 -- the anchored objective's own
     optimum never enters the favourable band (rate 0.46-2.6 over 24 cells)

    python scripts/theory_spectrum.py
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import erf

P, psi = norm.cdf, norm.pdf


def accs(c, eta, eps):
    """(clean, robust) accuracy in % of the linear readout c under an l_inf(eps) adversary."""
    n = np.linalg.norm(c)
    return P((c @ eta) / n) * 100, P((c @ eta - eps * np.abs(c).sum()) / n) * 100


def anchor_loss(c, eta, eps):
    """J(c) = E[(|R| + eps*||c||_1)^2], R = (c-eta).x  -- Theorem 1 generalized to D coords."""
    u = c - eta
    m, s = u @ eta, np.linalg.norm(u)
    if s < 1e-12:
        e1, er2 = abs(m), m * m
    else:
        e1 = s * np.sqrt(2 / np.pi) * np.exp(-m * m / (2 * s * s)) + abs(m) * erf(abs(m) / (s * np.sqrt(2)))
        er2 = m * m + s * s
    a = eps * np.abs(c).sum()
    return er2 + 2 * a * e1 + a * a


def spectrum(D=512, teacher_clean=0.774):
    """Linear reliability ramp, scaled so the natural readout has the teacher's clean accuracy."""
    eta = np.linspace(1.0, 0.02, D)
    return eta / np.linalg.norm(eta) * norm.ppf(teacher_clean)


def main():
    D, eps = 512, 0.05
    eta = spectrum(D)
    c_at = np.maximum(eta - eps, 0.0)
    n = np.linalg.norm(c_at)
    mc, mr = (c_at @ eta) / n, (c_at @ eta - eps * c_at.sum()) / n

    print(f"D={D} eps={eps}  AT keeps {int((eta > eps).sum())}/{D} coordinates")
    print("\n1. Prop. A2 exchange rate, restoring one deleted coordinate")
    print(f"{'coord':>6} {'eta/eps':>8} {'predicted':>10} {'measured':>10}")
    a0 = accs(c_at, eta, eps)
    for j in (70, 90, 150, 300):
        pred = eta[j] / (eps - eta[j]) * psi(mc) / psi(mr)
        c = c_at.copy(); c[j] += 1e-6
        a1 = accs(c, eta, eps)
        print(f"{j:6d} {eta[j]/eps:8.3f} {pred:10.2f} {(a1[0]-a0[0])/-(a1[1]-a0[1]):10.2f}")

    print("\n2. Frontier  c_lambda = (eta - lambda)_+   [lambda=eps is AT, lambda=0 the teacher]")
    print(f"{'lam/eps':>8} {'keep':>5} {'clean':>7} {'robust':>7} {'dclean':>8} {'drobust':>8} {'rate':>7}")
    for r in (1.0, 0.9, 0.8, 0.6, 0.35, 0.0):
        c = np.maximum(eta - r * eps, 0.0)
        a = accs(c, eta, eps)
        dc, dr = a[0] - a0[0], a[1] - a0[1]
        rate = dc / -dr if dr < -1e-9 else float("inf")
        print(f"{r:8.2f} {int((c>0).sum()):5d} {a[0]:7.2f} {a[1]:7.2f} {dc:+8.2f} {dr:+8.2f} {rate:7.1f}")

    print("\n3. Two-atom spectrum (T.1 geometry): 1 unperturbable + 511 bulk at eta/eps = 0.5")
    e2 = np.concatenate([[10.0], np.full(511, 0.5)])
    e2 = e2 / np.linalg.norm(e2) * norm.ppf(0.774)
    e2eps = e2[1] / 0.5
    for r in (1.0, 0.9, 0.7, 0.5, 0.0):
        a = accs(np.maximum(e2 - r * e2eps, 0.0), e2, e2eps)
        print(f"  lam/eps={r:.2f}  clean={a[0]:6.2f}  robust={a[1]:6.2f}")

    print("\n4. Population minimizer of the anchor loss -- overshoots, see 8.5")
    res = minimize(anchor_loss, x0=c_at, args=(eta, eps), method="L-BFGS-B",
                   bounds=[(0, None)] * D, options=dict(maxiter=50000, ftol=1e-16, gtol=1e-14))
    a = accs(res.x, eta, eps)
    print(f"  anchor  keeps {int((res.x>1e-10).sum()):3d}/{D}  clean={a[0]:6.2f} robust={a[1]:6.2f}")
    print(f"  AT      keeps {int((c_at>0).sum()):3d}/{D}  clean={a0[0]:6.2f} robust={a0[1]:6.2f}")

    print("\n5. rate universality at s=0.1 (lambda = 0.9 eps): rate is flat, the gain is not")
    rng = np.random.default_rng(0)
    shapes = {"linear ramp": np.linspace(1, 0.02, D), "1/j": 1 / np.arange(1, D + 1),
              "lognormal(0,1)": np.sort(rng.lognormal(0, 1, D))[::-1],
              "uniform": np.sort(rng.uniform(0, 1, D))[::-1]}
    for nm, e in shapes.items():
        e = e / np.linalg.norm(e) * norm.ppf(0.774)
        ee = np.quantile(e, 0.875)
        b, a = accs(np.maximum(e - ee, 0), e, ee), accs(np.maximum(e - 0.9 * ee, 0), e, ee)
        dc, dr = a[0] - b[0], a[1] - b[1]
        print(f"  {nm:<16} dclean {dc:+6.2f}  drobust {dr:+6.2f}  rate {dc/-dr:6.1f}")

    print("\n6. Prop. A1 under the adversarial-logistic surrogate")
    gx, gw = np.polynomial.hermite_e.hermegauss(81); gw = gw / gw.sum()
    def advlog(w):
        return float(gw @ np.logaddexp(0, eps * np.abs(w).sum() - w @ eta - np.linalg.norm(w) * gx))
    w = minimize(advlog, x0=c_at, method="Powell",
                 options=dict(maxiter=200000, xtol=1e-10, ftol=1e-12)).x
    w = np.where(np.abs(w) < 1e-6, 0, w)
    print(f"  support advCE={int((w!=0).sum())}  hard-threshold={int((eta>eps).sum())}  "
          f"corr={np.corrcoef(w, c_at)[0,1]:.4f}")

    print("\n7. Prop. C: the clean-robust margin gap is exactly eps * ||c||_1/||c||_2")
    for nm, c in (("1-sparse", np.eye(D)[0]), ("64 equal", np.r_[np.ones(64), np.zeros(D-64)]),
                  ("all equal", np.ones(D))):
        print(f"  {nm:<10} ||c||_1/||c||_2 = {np.abs(c).sum()/np.linalg.norm(c):6.2f}")


if __name__ == "__main__":
    main()
