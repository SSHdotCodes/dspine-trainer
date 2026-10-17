import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from serving.oracle import FrozenOracle
from serving.profile.joint_exactness import joint_distribution,pooled_fit


def fixture(tmp_path):
    nodes=[]
    for path in [[],[0],[1],[0,0],[0,1],[1,0],[1,1]]:
        weights=[.7,.3] if not path or path[-1]==0 else [.2,.8]
        nodes.append(dict(path=path,ids=[0,1],probabilities=weights))
    prefix=dict(name='toy',input_ids=[44,55],nodes=nodes)
    manifest=dict(protocol=dict(joint_tokens=3),prefixes=[prefix])
    path=tmp_path/'oracle.json';path.write_text(json.dumps(manifest))
    return path,hashlib.sha256(path.read_bytes()).hexdigest(),prefix


def test_frozen_oracle_conditions_on_actual_prefix_and_cleans_up(tmp_path):
    path,sha,prefix=fixture(tmp_path)
    oracle=FrozenOracle(path,sha,'cpu')
    req=SimpleNamespace(rid='r1',origin_input_ids=[44,55],
        sampling_params=SimpleNamespace(max_new_tokens=3,ignore_eos=True))
    anchor,_=oracle.prefill([req]);a=int(anchor[0])
    p=torch.zeros(1,3,7)
    candidates=torch.tensor([[a,1,6]])
    oracle.override(p,candidates,[req])
    assert torch.allclose(p[0,0,:2],torch.tensor([.7,.3] if a==0 else [.2,.8]))
    assert torch.allclose(p[0,1,:2],torch.tensor([.2,.8]))
    # Simulate a rejection producing0, then a new round anchored on that token.
    oracle.advance([req],torch.tensor([[0,6,6]]),torch.tensor([1]))
    oracle.override(p,torch.tensor([[0,6,6]]),[req])
    assert torch.allclose(p[0,0,:2],torch.tensor([.7,.3]))
    oracle.advance([req],torch.tensor([[1,6,6]]),torch.tensor([1]))
    assert not oracle.histories and not oracle.names
    with pytest.raises(ValueError,match='hash mismatch'):FrozenOracle(path,'wrong','cpu')


def test_analytic_joint_gof_detects_bias(tmp_path):
    _,_,prefix=fixture(tmp_path);expected=joint_distribution(prefix,3)
    keys=sorted(expected);p=np.array([expected[k] for k in keys]);p/=p.sum()
    rng=np.random.default_rng(217);n=50000
    counts=dict(zip(keys,rng.multinomial(n,p)))
    result=pooled_fit(counts,expected,n,rng,simulations=2000,bootstrap=100)
    assert result['g_monte_carlo_p']>.01
    # Deliberately bias the complete joint, preserving normalization.
    counts[keys[0]]+=3000;counts[keys[-1]]-=3000
    result=pooled_fit(counts,expected,n,rng,simulations=2000,bootstrap=100)
    assert result['g_monte_carlo_p']<.001


def test_joint3_uniform_tail_keeps_power_instead_of_one_bucket():
    # At20^3 outcomes every expected leaf is6.25 at N50K. Pooling all rare
    # leaves together would produce one bin and never detect any discrepancy.
    expected={(i,):1/8000 for i in range(8000)}
    counts={key:7 if key[0]<2000 else 6 for key in expected}
    result=pooled_fit(counts,expected,50000,np.random.default_rng(13),simulations=100,bootstrap=10)
    assert result['pooled_bins']==4000
    assert result['minimum_expected_count']>=10
