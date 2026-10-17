"""Frozen-oracle joint goodness of fit; native path comparisons are descriptive."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import islice
import json
from pathlib import Path
import time

import numpy as np
import requests
from scipy import stats


def joint_distribution(prefix,length):
    nodes={tuple(row['path']):row for row in prefix['nodes']}
    result={():1.}
    for _ in range(length):
        expanded={}
        for path,mass in result.items():
            row=nodes[path];p=np.array(row['probabilities'],dtype=np.float64);p/=p.sum()
            for token,weight in zip(row['ids'],p):expanded[(*path,token)]=mass*weight
        result=expanded
    return result


def pooled_fit(counts,expected,n,rng,simulations=20000,bootstrap=2000):
    if set(counts)-set(expected):raise ValueError('observed zero-probability joint outcome')
    keys=sorted(expected);p=np.array([expected[k] for k in keys]);p/=p.sum()
    observed=np.array([counts[k] for k in keys]);assert observed.sum()==n
    large=p*n>=10
    bins=[[i] for i in np.flatnonzero(large)]
    rare=[];mass=0.
    # Group rare leaves in canonical path order. A single all-tail bucket would
    # make uniform20^3 at N50K a one-bin test with no power at all.
    rare_bins=[]
    for i in np.flatnonzero(~large):
        rare.append(int(i));mass+=p[i]
        if mass*n>=10:
            rare_bins.append(rare);rare=[];mass=0.
    if rare:
        if rare_bins:rare_bins[-1].extend(rare)
        elif bins:
            smallest=min(range(len(bins)),key=lambda b:p[bins[b][0]])
            bins[smallest].extend(rare)
        else:raise ValueError('too few samples for even one expected-count10 bin')
    bins.extend(rare_bins)
    bp=np.array([p[index].sum() for index in bins])
    bo=np.array([observed[index].sum() for index in bins])
    bp/=bp.sum();exp=n*bp
    def g(x):return 2*np.sum(np.where(x>0,x*np.log(np.maximum(x,1)/exp),0),axis=-1)
    observed_g=float(g(bo));pearson=float(np.sum((bo-exp)**2/exp))
    null_g=[];null_tv=[];tv_boot=[]
    empirical=observed/n;tv=float(.5*np.abs(empirical-p).sum())
    for start in range(0,simulations,256):
        draw=rng.multinomial(n,bp,size=min(256,simulations-start));null_g.extend(g(draw))
    for start in range(0,bootstrap,64):
        size=min(64,bootstrap-start)
        draw=rng.multinomial(n,empirical,size=size);tv_boot.extend(.5*np.abs(draw/n-p).sum(-1))
        draw=rng.multinomial(n,p,size=size);null_tv.extend(.5*np.abs(draw/n-p).sum(-1))
    return dict(samples=n,joint_support=len(keys),pooled_bins=len(bp),pooled_tail_probability=float(p[~large].sum()),
        minimum_expected_count=float(exp.min()),tail_pool_rule='canonical rare-leaf groups expected>=10; final remainder merged',g_statistic=observed_g,
        g_monte_carlo_p=float((1+np.count_nonzero(np.asarray(null_g)>=observed_g))/(simulations+1)),
        pearson=pearson,pearson_asymptotic_p=float(stats.chi2.sf(pearson,len(bp)-1)),
        total_variation=tv,tv_bootstrap_percentile95=np.quantile(tv_boot,[.025,.975]).tolist(),
        tv_null_sampling_percentile95=np.quantile(null_tv,[.025,.975]).tolist(),
        tv_note='Plug-in TV includes multinomial sampling noise; bootstrap intervals are descriptive, not a test of TV=0.')


def collect(a):
    if not 30200<=a.port<=30299:raise ValueError('isolated port required')
    manifest=json.loads(a.oracle.read_text());base=f'http://127.0.0.1:{a.port}'
    deadline=time.monotonic()+180
    while True:
        try:
            # /health generates an internal greedy probe. Oracle mode accepts
            # only its registered T1/three-token prefixes, so readiness must
            # inspect metadata without submitting an unrelated generation.
            if requests.get(base+'/server_info',timeout=2).status_code==200:break
        except requests.RequestException:pass
        if time.monotonic()>deadline:raise TimeoutError('joint-test server readiness timeout')
        time.sleep(1)
    prefixes=[r for r in manifest['prefixes'] if not a.prefix or r['name']==a.prefix]
    if not prefixes:raise ValueError('unknown prefix')
    info=requests.get(base+'/server_info',timeout=30);info.raise_for_status()
    a.output.with_suffix('.server-info.json').write_text(json.dumps(info.json(),indent=2)+'\n')
    def run(group):
        inputs=[row['input_ids'] for row,index in group]
        response=requests.post(base+'/generate',json=dict(input_ids=inputs if a.http_batch else inputs[0],return_logprob=True,
            sampling_params=dict(temperature=1.,top_k=20,top_p=.95,min_p=0.,ignore_eos=True,max_new_tokens=3)),timeout=(30,300))
        response.raise_for_status();values=response.json()
        if not a.http_batch:values=[values]
        if len(values)!=len(group):raise ValueError('unexpected response batch length')
        result=[]
        for (row,index),value in zip(group,values):
            meta=value['meta_info']
            if bool(meta.get('spec_verify_ct',0))!=(a.mode!='ordinary'):raise ValueError('wrong speculative mode')
            ids=value.get('output_ids') or [r[1] for r in meta['output_token_logprobs']]
            if len(ids)!=3:raise ValueError('truncated joint sample')
            result.append(dict(prefix=row['name'],sample=index,ids=ids,verify_rounds=meta.get('spec_verify_ct',0),mode=a.mode,
                               http_batch=len(group)))
        return result
    protocol_path=a.output.with_suffix('.protocol.json')
    protocol=dict(mode=a.mode,oracle_sha256=hashlib.sha256(a.oracle.read_bytes()).hexdigest())
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise ValueError('resume protocol differs')
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    completed=set()
    if a.output.exists():
        completed={(r['prefix'],r['sample']) for r in map(json.loads,a.output.read_text().splitlines())}
    jobs=((r,i) for r in prefixes for i in range(manifest['protocol']['samples_per_prefix']) if (r['name'],i) not in completed)
    def groups():
        while group:=list(islice(jobs,a.batch_size if a.http_batch else 1)):yield group
    count=0
    with a.output.open('a') as stream,ThreadPoolExecutor(a.client_workers or a.batch_size) as pool:
        for group in pool.map(run,groups()):
            for row in group:
                stream.write(json.dumps(row)+'\n');stream.flush();count+=1
                if count%1000==0:print(json.dumps(dict(new_samples=count,resumed_samples=len(completed))),flush=True)


def analyze(a):
    manifest=json.loads(a.oracle.read_text());protocol=manifest['protocol']
    rows=[json.loads(line) for path in a.samples for line in path.read_text().splitlines()]
    if any(r.get('mode')!='oracle-spec' for r in rows):raise ValueError('primary fit requires explicit frozen-oracle mode')
    identities=[(r['prefix'],r['sample']) for r in rows]
    if len(set(identities))!=len(identities):raise ValueError('duplicate draws')
    results=[];rng=np.random.default_rng(protocol['analysis_seed'])
    for prefix in manifest['prefixes']:
        samples=[r for r in rows if r['prefix']==prefix['name']]
        if len(samples)!=protocol['samples_per_prefix']:raise ValueError('pre-registered sample count not reached')
        for length in (2,3):
            counts=Counter(tuple(r['ids'][:length]) for r in samples)
            result=pooled_fit(counts,joint_distribution(prefix,length),len(samples),rng,
                simulations=protocol['null_simulations'],bootstrap=protocol['bootstrap_simulations'])
            results.append(dict(prefix=prefix['name'],joint_tokens=length,**result))
    threshold=protocol['family_alpha']/(2*len(manifest['prefixes']))
    for row in results:row['mismatch_detected']=row['g_monte_carlo_p']<threshold
    receipt=dict(oracle_sha256=hashlib.sha256(a.oracle.read_bytes()).hexdigest(),protocol=protocol,
        threshold_each=threshold,results=results,mismatch_detected=any(r['mismatch_detected'] for r in results),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.samples})
    a.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('collect');q.add_argument('--port',type=int,required=True);q.add_argument('--oracle',type=Path,required=True)
    q.add_argument('--prefix');q.add_argument('--mode',choices=('oracle-spec','native-spec','ordinary'),required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--batch-size',type=int,choices=(1,2),default=2)
    q.add_argument('--http-batch',action='store_true',help='send paired inputs in one HTTP request for bs2 scheduling')
    q.add_argument('--client-workers',type=int,choices=range(1,33),help='HTTP queue depth; server still limits active requests to2')
    q=sub.add_parser('analyze');q.add_argument('--oracle',type=Path,required=True);q.add_argument('--samples',type=Path,nargs='+',required=True)
    q.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    (collect if a.command=='collect' else analyze)(a)


if __name__=='__main__':main()
