import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier

EPOCHS, LR, TAU0, TAUE, TAUG, GAMMA, RHO = 500, 3e-3, 0.7, 0.03, 0.98, 2.0, 0.1
N_CONCEPTS_H3 = 32


class Net(nn.Module):
    def __init__(self, X, y, n_cls, cfg):
        super().__init__()
        F = X.shape[1]
        thr = np.median(X, 0).copy()
        for j in range(F):
            t = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X[:, j:j + 1], y).tree_.threshold[0]
            if t > -2:
                thr[j] = t
        self.t = nn.Parameter(torch.tensor(thr, dtype=torch.float32))
        self.kind = cfg['model']
        h1 = {'H2': cfg['rules'], 'H3': N_CONCEPTS_H3}[self.kind]
        self.W1 = nn.Parameter(torch.empty(h1, F))
        nn.init.xavier_uniform_(self.W1)
        last = h1
        if self.kind != 'H2':
            self.W2 = nn.Parameter(torch.empty(cfg['rules'], h1))
            nn.init.xavier_uniform_(self.W2)
            last = cfg['rules']
        self.Wo = nn.Parameter(torch.empty(n_cls, last))
        nn.init.xavier_uniform_(self.Wo)
        self.b = nn.Parameter(torch.zeros(n_cls))

    def and_ws(self):
        return [self.W1] + ([self.W2] if self.kind != 'H2' else [])

    def forward(self, x, tau):
        a = torch.sigmoid(torch.sigmoid(GAMMA * (x - self.t) / tau) @ self.W1.T / tau)
        if self.kind != 'H2':
            a = torch.sigmoid(a @ self.W2.T / tau)
        return a @ self.Wo.T + self.b

    def discrete(self, x, rho=RHO):
        with torch.no_grad():
            def prune(W):
                return (W.abs() >= rho * W.abs().max(1, keepdim=True).values) & (W.abs() > 0)
            s = (x > self.t).float()
            k1 = prune(self.W1)
            c = ((s @ (self.W1 * k1).T) > 0).float()
            if self.kind == 'H2':
                alive = (self.Wo.abs().max(0).values >= rho * self.Wo.abs().max()) & k1.any(1)
                a = c * alive
                lens = k1[alive].sum(1)
                dl = int(lens.sum())
                feats = set(torch.where(k1[alive].any(0))[0].tolist())
                mean_len = float(lens.float().mean()) if alive.any() else 0.0
                info = {
                    'dl': dl,
                    'n_rules': int(alive.sum()),
                    'items_per_rule': mean_len,
                    'expanded_len': mean_len,
                    'feats': feats,
                }
                return a @ self.Wo.T + self.b, info
            k2 = prune(self.W2) & k1.any(1)[None]
            r = ((c @ (self.W2 * k2).T) > 0).float()
            alive = (self.Wo.abs().max(0).values >= rho * self.Wo.abs().max()) & k2.any(1)
            used_c = k2[alive].any(0)
            conc_len = k1.sum(1)
            dl = int(conc_len[used_c].sum() + k2[alive].sum())
            exp_len = [(conc_len[k2[i]]).sum().item() for i in torch.where(alive)[0]]
            feats = set(torch.where(k1[used_c].any(0))[0].tolist()) if used_c.any() else set()
            info = {
                'dl': dl,
                'n_rules': int(alive.sum()),
                'n_concepts': int(used_c.sum()),
                'items_per_rule': float(k2[alive].sum(1).float().mean()) if alive.any() else 0.0,
                'expanded_len': float(np.mean(exp_len)) if exp_len else 0.0,
                'feats': feats,
            }
            return (r * alive) @ self.Wo.T + self.b, info


def train(cfg, X_tr, y_tr, n_cls, seed, epochs=EPOCHS):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = Net(X_tr, y_tr, n_cls, cfg)
    Xt, yt = torch.tensor(X_tr), torch.tensor(y_tr)
    cnt = np.bincount(y_tr, minlength=n_cls)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(y_tr) / (n_cls * cnt), dtype=torch.float32), label_smoothing=0.05)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    lam = cfg['lam']
    for ep in range(epochs):
        tau = TAUE + (TAU0 - TAUE) * TAUG ** ep
        opt.zero_grad()
        loss = lossf(m(Xt, tau), yt) + lam * (sum(W.abs().sum() for W in m.and_ws()) + m.Wo.abs().sum())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for W in m.and_ws() + [m.Wo]:
                W.data = torch.sign(W.data) * (W.data.abs() - lam * LR).clamp(min=0)
        sch.step()
    return m


def parts_of(m, rho=RHO):
    with torch.no_grad():
        def prune(W):
            return (W.abs() >= rho * W.abs().max(1, keepdim=True).values) & (W.abs() > 0)
        k1 = prune(m.W1)
        W1 = m.W1 * k1
        if m.kind == 'H2':
            alive = (m.Wo.abs().max(0).values >= rho * m.Wo.abs().max()) & k1.any(1)
            layers = [W1[alive].numpy().astype(np.float64)]
        else:
            k2 = prune(m.W2) & k1.any(1)[None]
            alive = (m.Wo.abs().max(0).values >= rho * m.Wo.abs().max()) & k2.any(1)
            layers = [W1.numpy().astype(np.float64), (m.W2 * k2)[alive].numpy().astype(np.float64)]
        return m.t.numpy(), layers, m.Wo[:, alive].numpy().astype(np.float64), m.b.numpy().astype(np.float64)
