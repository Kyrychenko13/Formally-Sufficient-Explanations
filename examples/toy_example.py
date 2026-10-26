import itertools, json, os
import numpy as np

W = np.array([[2., 1., -1.], [-1., 0., 2.]]); V = np.array([[1., 0.], [0., 1.]]); b = np.array([0., 0.2])
s = np.array([1., 1., 0.]); NAMES = ['A', 'B']


def decide(x):
    a = ((W @ x) > 0).astype(float); z = V @ a + b
    return int(np.argmax(z)), a, z


def bounds(fixed):
    lo = np.where(fixed, s, 0.); hi = np.where(fixed, s, 1.)
    Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
    L, U = Wp @ lo + Wn @ hi, Wp @ hi + Wn @ lo
    l, u = (L > 0).astype(float), (U > 0).astype(float)
    c_hat = decide(s)[0]; margins = {}
    for c in range(len(V)):
        if c == c_hat: continue
        d = V[c_hat] - V[c]
        margins[NAMES[c]] = float(np.minimum(d * l, d * u).sum() + b[c_hat] - b[c])
    return L, U, l, u, margins


c_hat, a, z = decide(s)
fixed = np.ones(3, bool); imp = np.abs(W).sum(0); steps = []
for j in np.argsort(imp, kind='stable'):
    fixed[j] = False
    L, U, l, u, m = bounds(fixed)
    ok = all(v > 0 for v in m.values())
    free = np.where(~fixed)[0]; exact = True
    for vals in itertools.product([0., 1.], repeat=len(free)):
        x = s.copy(); x[free] = vals
        if decide(x)[0] != c_hat: exact = False
    steps.append({'freed': f's{j + 1}', 'L': L.round(3).tolist(), 'U': U.round(3).tolist(),
                  'state': ['1' if li else ('0' if not ui else '?') for li, ui in zip(l, u)],
                  'margin_vs_B': round(m['B'], 3), 'removed': bool(ok), 'exact_sufficient': exact})
    if not ok: fixed[j] = True
cube = {''.join(str(int(v)) for v in x): NAMES[decide(np.array(x, float))[0]] for x in itertools.product([0, 1], repeat=3)}
out = {'instance': s.astype(int).tolist(), 'decision': NAMES[c_hat], 'a': a.tolist(), 'z': z.round(3).tolist(),
       'order_importance': imp.tolist(), 'steps': steps,
       'explanation': {f's{j + 1}': int(s[j]) for j in np.where(fixed)[0]}, 'cube_decisions': cube}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toy_example.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
