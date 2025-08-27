import numpy as np

def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order, start=1):
        adj[idx] = max(prev, (m - rank + 1) * pvals[idx])
        prev = adj[idx]
    return np.minimum(adj, 1.0).tolist()
