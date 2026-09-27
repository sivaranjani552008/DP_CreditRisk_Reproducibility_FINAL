"""Cross-check of src/dp_ppnn.py per-example clipped gradients and Adam against PyTorch (requires torch)."""
import sys, numpy as np, torch
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src')); import dp_ppnn as P
torch.set_default_dtype(torch.float64)
rng=np.random.default_rng(0)
for arch in ['report','keras_literal']:
    m=P.MLP(12,P.ARCHS[arch],rng)
    X=rng.normal(size=(7,12)); y=rng.integers(0,2,7).astype(float)
    # torch per-example
    Ws=[torch.tensor(w,requires_grad=True) for w in m.W]; bs=[torch.tensor(b,requires_grad=True) for b in m.b]
    per=[]
    for i in range(7):
        h=torch.tensor(X[i:i+1])
        for l,(W,b) in enumerate(zip(Ws,bs)):
            h=h@W+b
            if l<len(Ws)-1: h=torch.relu(h)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(h[:,0],torch.tensor(y[i:i+1]))
        g=torch.autograd.grad(loss,[p for pair in zip(Ws,bs) for p in pair])
        per.append(torch.cat([x.reshape(-1) for x in g]).numpy())
    per=np.array(per)
    for clip in ['l1','l2']:
        nrm=np.abs(per).sum(1) if clip=='l1' else np.linalg.norm(per,axis=1)
        C=0.5; f=np.minimum(1,C/nrm); ref=(per*f[:,None]).mean(0)
        out,norms=m.clipped_mean_grads(X,y,clip,C)
        mine=np.concatenate([g.reshape(-1) for g in out[0]])
        print(arch,clip,'norm err',np.abs(norms-nrm).max(),'grad err',np.abs(mine-ref).max(), 'd',m.n_params)
    # shards
    out,_=m.clipped_mean_grads(X,y,'l2',0.5,[np.arange(3),np.arange(3,7)])
    ref=(per[3:]*np.minimum(1,0.5/np.linalg.norm(per[3:],axis=1))[:,None]).mean(0)
    print('shard err',np.abs(np.concatenate([g.reshape(-1) for g in out[1]])-ref).max())
# Adam vs torch
p=[rng.normal(size=(3,2))]; pt=torch.tensor(p[0].copy(),requires_grad=True)
opt=P.Adam(p,0.01); topt=torch.optim.Adam([pt],lr=0.01)
for t in range(50):
    g=rng.normal(size=(3,2)); opt.step(p,[g]); pt.grad=torch.tensor(g); topt.step()
print('adam err',np.abs(p[0]-pt.detach().numpy()).max())
