import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N_VERIFY = 200


def predict(P, S):
    _, layers, Wo, b = P
    a = S
    for W in layers:
        a = ((a @ W.T) > 0).astype(float)
    return (a @ Wo.T + b).argmax(1)


def sufficient_bounds(P, s, fixed, c_hat):
    _, layers, Wo, b = P
    lo = np.where(fixed, s, 0.0)
    hi = np.where(fixed, s, 1.0)
    for W in layers:
        Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
        zlo, zhi = lo @ Wp.T + hi @ Wn.T, hi @ Wp.T + lo @ Wn.T
        lo, hi = (zlo > 0).astype(float), (zhi > 0).astype(float)
    for c in range(Wo.shape[0]):
        if c == c_hat:
            continue
        d = Wo[c_hat] - Wo[c]
        if (np.minimum(d * lo, d * hi)).sum() + b[c_hat] - b[c] <= 0:
            return False
    return True


def sufficient_milp(P, s, fixed, c_hat):
    _, layers, Wo, b = P
    free = np.where(~fixed)[0]
    nf = len(free)
    nv = nf + sum(W.shape[0] for W in layers)
    for c in range(Wo.shape[0]):
        if c == c_hat:
            continue
        A, lb, ub = [], [], []
        prev_idx = None
        off = nf
        for li, W in enumerate(layers):
            for r in range(W.shape[0]):
                row = np.zeros(nv)
                const = 0.0
                if li == 0:
                    const = float((W[r] * s)[fixed].sum())
                    row[:nf] = W[r][free]
                else:
                    row[prev_idx] = W[r]
                M = np.abs(W[r]).sum() + abs(const) + 1.0
                a = off + r
                r1 = row.copy()
                r1[a] = -M
                A.append(r1)
                lb.append(-np.inf)
                ub.append(-const)
                r2 = row.copy()
                r2[a] = -M
                A.append(r2)
                lb.append(1e-6 - M - const)
                ub.append(np.inf)
            prev_idx = np.arange(off, off + W.shape[0])
            off += W.shape[0]
        d = Wo[c] - Wo[c_hat]
        obj = np.zeros(nv)
        obj[prev_idx] = -d
        res = milp(obj, constraints=LinearConstraint(np.array(A), lb, ub), integrality=np.ones(nv),
                   bounds=Bounds(0, 1), options={'presolve': False})
        if res.status != 0:
            return False
        if -res.fun + b[c] - b[c_hat] >= 0:
            return False
    return True


def explain(P, s, check=sufficient_bounds):
    _, layers, Wo, b = P
    c_hat = int(predict(P, s[None])[0])
    imp = np.abs(layers[0]).sum(0)
    fixed = imp > 0
    for j in np.argsort(imp):
        if not fixed[j]:
            continue
        fixed[j] = False
        if not check(P, s, fixed, c_hat):
            fixed[j] = True
    return c_hat, fixed


def violations(P, s, keep, c_hat, S_tr, rng, n_verify=N_VERIFY):
    free = ~keep
    R = np.repeat(s[None], 2 * n_verify, 0)
    R[:n_verify][:, free] = rng.randint(0, 2, (n_verify, free.sum()))
    R[n_verify:][:, free] = S_tr[rng.randint(0, len(S_tr), n_verify)][:, free]
    return bool((predict(P, R) != c_hat).any())


def conditions(P, s, fixed):
    t = P[0]
    return [(int(j), '>' if s[j] == 1 else '<=', float(t[j])) for j in np.where(fixed)[0]]
