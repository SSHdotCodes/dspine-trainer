"""Randomized PIT from exact sparse target rows; independent NumPy RNG.

Outcome-conditioned accepted/correction groups do NOT have a uniform null.
Reached row indices and the aggregate do. Raw chunks preserve every p, id and V.
"""
import atexit
import json
from pathlib import Path

import numpy as np

KINDS = {0: 'ordinary-or-prefill', 1: 'accepted-draft', 2: 'correction', 3: 'bonus'}


def randomized_pit(ids, probabilities, token, uniform):
    ids = np.asarray(ids, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    keep = probabilities > 0
    ids, probabilities = ids[keep], probabilities[keep]
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('duplicate sparse support ids')
    total = probabilities.sum()
    if not np.isfinite(total) or abs(total-1.) > 1e-4:
        raise ValueError(f'target row is not normalized: {total}')
    probabilities = probabilities / total
    order = np.lexsort((ids, -probabilities))
    ids, probabilities = ids[order], probabilities[order]
    found = np.flatnonzero(ids == token)
    if len(found) != 1:
        raise ValueError(f'emitted token {token} has zero target probability')
    i = int(found[0])
    before, selected = float(probabilities[:i].sum()), float(probabilities[i])
    return before + uniform*selected, before, selected


class PitWriter:
    def __init__(self, root, *, seed, label, chunk_size=512):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)
        self.label, self.chunk_size, self.chunk = label, chunk_size, 0
        self.rows = []
        self.count = 0
        metadata=self.root/f'{label}-metadata.json'
        if metadata.exists():raise FileExistsError(metadata)
        metadata.write_text(json.dumps(dict(
            seed=seed, rng='numpy.PCG64 independent of torch/native sampler',
            order='descending target probability, ties ascending token id', kinds=KINDS,
            normalization='float64 division by sum of logged float32 weights',
            tests='aggregate/reached-row uniformity; outcome-kind groups descriptive'), indent=2)+'\n')
        atexit.register(self.flush)

    def add(self, ids, probabilities, tokens, rows, kinds):
        for support, probs, token, row, kind in zip(ids, probabilities, tokens, rows, kinds):
            v = float(self.rng.random())
            u, before, selected = randomized_pit(support, probs, int(token), v)
            self.rows.append((np.asarray(support,dtype=np.int32), np.asarray(probs,dtype=np.float32),
                              int(token), int(row), int(kind), v, u, before, selected))
            self.count += 1
            if len(self.rows) >= self.chunk_size:
                self.flush()

    def flush(self):
        if not self.rows:
            return
        rows, self.rows = self.rows, []
        columns = list(zip(*rows))
        names = ['ids','probabilities','token','row','kind','v','u','f_minus','selected_probability']
        data = {key:np.asarray(value) for key,value in zip(names,columns)}
        path = self.root/f'{self.label}-{self.chunk:06d}.npz'
        if path.exists():
            raise FileExistsError(path)
        temporary=path.with_suffix('.tmp')
        with temporary.open('xb') as stream:
            np.savez_compressed(stream, **data)
        temporary.rename(path)
        self.chunk += 1


def uniform_ad(values):
    values = np.sort(np.asarray(values,dtype=np.float64))
    values = np.clip(values, np.finfo(float).eps, 1-np.finfo(float).eps)
    n = len(values)
    return float(-n-np.mean((2*np.arange(1,n+1)-1)*(np.log(values)+np.log1p(-values[::-1]))))


def ad_null(seed=170263, simulations=100000, components=1024):
    """Known-uniform AD limiting law: sum_j chi2(1)/(j(j+1)).

    Omitted tail is replaced by its exact mean1/(components+1); its standard
    deviation is <2.5e-5 for1024 components. P-values are Monte Carlo estimates,
    not the normal-distribution Anderson-Darling table used by scipy.anderson.
    """
    rng = np.random.default_rng(seed)
    j = np.arange(1,components+1,dtype=np.float64)
    weight = 1/(j*(j+1))
    result = []
    for start in range(0,simulations,1024):
        z = rng.standard_normal((min(1024,simulations-start), components))
        result.extend(np.sum((z*z)*weight,axis=1)+1/(components+1))
    return np.sort(result)


def uniform_tests(values, null):
    from scipy import stats
    values = np.asarray(values)
    ks = stats.kstest(values, 'uniform', method='asymp')
    ad = uniform_ad(values)
    lags = np.arange(1,17)
    if len(values) < 100:
        return dict(count=len(values),ks=float(ks.statistic),ks_p=float(ks.pvalue),ad=ad,
                    ad_p=float((1+len(null)-np.searchsorted(null,ad))/(len(null)+1)))
    centered = values-values.mean()
    covariance = np.dot(centered,centered)
    correlations = np.array([np.dot(centered[:-lag],centered[lag:])/covariance for lag in lags]) if covariance else np.ones(16)
    q = len(values)*(len(values)+2)*np.sum(correlations**2/(len(values)-lags))
    return dict(count=len(values),ks=float(ks.statistic),ks_p=float(ks.pvalue),ad=ad,
                ad_p=float((1+len(null)-np.searchsorted(null,ad))/(len(null)+1)),
                lag1_correlation=float(correlations[0]),ljung_box16=float(q),ljung_box16_p=float(stats.chi2.sf(q,16)))
