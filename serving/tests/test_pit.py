import numpy as np
from scipy.stats import kstest

from serving.pit import PitWriter, randomized_pit, uniform_ad


def test_pit_sparse_support_ties_and_receipt(tmp_path):
    # Canonical order is .5(id8),.25(id2),.25(id9), regardless of storage order.
    u,before,p=randomized_pit([9,8,2,-1],[.25,.5,.25,0],9,.4)
    assert (before,p,u)==(.75,.25,.85)
    writer=PitWriter(tmp_path,seed=17,label='test',chunk_size=2)
    writer.add([[9,8,2]]*2,[[.25,.5,.25]]*2,[9,8],[1,2],[1,2])
    with np.load(tmp_path/'test-000000.npz') as receipt:
        for i in range(2):
            assert randomized_pit(receipt['ids'][i],receipt['probabilities'][i],receipt['token'][i],receipt['v'][i])[0]==receipt['u'][i]


def test_pit_acceptance_conditioned_groups_are_not_uniform():
    # Exact rejection from q=(.1,.9) to p=(.7,.3), including residual correction.
    rng=np.random.default_rng(1907);n=100000
    y=(rng.random(n)>=.1).astype(int)
    accept=rng.random(n)<np.minimum(1,np.array([.7,.3])[y]/np.array([.1,.9])[y])
    y[~accept]=0  # residual=(.6,0), always id0
    p=np.array([.7,.3]);v=rng.random(n)
    u=np.where(y==0,0.,.7)+v*p[y]
    assert kstest(u,'uniform').pvalue>.01
    assert kstest(u[accept],'uniform').statistic>.4
    assert kstest(u[~accept],'uniform').statistic>.29
    assert uniform_ad(u)<4
