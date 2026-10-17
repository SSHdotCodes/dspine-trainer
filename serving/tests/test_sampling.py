import torch
from serving.sampling import process_logits, verify_sparse, walk_cache


def test_native_joint_filters_and_cutoff_ties():
    flat=process_logits(torch.zeros(29),1.,20,.95)
    torch.testing.assert_close(flat,torch.full((29,),1/29))
    # Joint filtering retains all top20 here; top-k-first nucleus drops token19.
    p=process_logits(torch.linspace(.1,0,29),1.,20,.95)
    assert (p[:20]>0).all() and (p[20:]==0).all()
    logits=torch.full((29,),-1000.)
    logits[:3]=torch.tensor([.5,.25,.25]).log()
    p=process_logits(logits,1.,20,.6)
    torch.testing.assert_close(p[:3],torch.tensor([.5,.25,.25]))


def test_t1_fixed_prefix_distribution():
    # Sparse proposal deliberately misses high-mass target tokens. This catches
    # residual correction that subtracts only the chosen token's q probability.
    torch.manual_seed(41); n=100_000; vocab=29
    p=process_logits(torch.linspace(-2,2,vocab),1.,20,.95)
    ids=torch.tensor([0,3,7,11,15,19,23,26])
    q=torch.tensor([.02,.06,.09,.16,.08,.20,.30,.09])
    chosen=torch.multinomial(q,n,replacement=True)
    proposals=ids[chosen,None]
    target=p.expand(n,2,vocab)
    result=verify_sparse(proposals,ids.expand(n,1,-1),q.expand(n,1,-1),target)
    freq=torch.bincount(result.output_ids[:,0],minlength=vocab)/n
    error=(freq-p).abs()
    threshold=6*torch.sqrt(p*(1-p)/n)+2/n
    assert (error<=threshold).all(),(freq,p)
    assert (freq[p==0]==0).all()


def test_greedy_and_all_accepted_bonus():
    proposals=torch.tensor([[2,3],[1,2]])
    ids=proposals[...,None]; q=torch.ones(2,2,1)
    p=torch.nn.functional.one_hot(torch.tensor([[2,3,4],[0,2,4]]),5).float()
    r=verify_sparse(proposals,ids,q,p,torch.zeros(2,2),torch.tensor([.4,.8]))
    assert r.commit_lens.tolist()==[3,1]
    assert r.output_ids[0].tolist()==[2,3,4]
    assert r.output_ids[1,0]==0


def test_transition_walk_q_is_realized_conditional_row():
    ids=torch.tensor([[[1,2],[3,4],[5,6]]])
    initial=torch.tensor([[[0.,1.],[0.,0.],[0.,0.]]])
    edges=torch.tensor([[[[3.,0.],[0.,3.]],[[0.,2.],[2.,0.]]]])
    tokens,q=walk_cache(ids,initial,edges,temperature=1.,uniforms=torch.tensor([[.5,.5,.5]]))
    assert tokens.tolist()==[[2,4,5]]
    torch.testing.assert_close(q[0,1],edges[0,0,1].softmax(-1))
    torch.testing.assert_close(q[0,2],edges[0,1,1].softmax(-1))
    tokens,q=walk_cache(ids,initial,edges,temperature=0.)
    assert tokens.tolist()==[[2,4,5]]
    assert q.tolist()==[[[0.,1.],[0.,1.],[1.,0.]]]
