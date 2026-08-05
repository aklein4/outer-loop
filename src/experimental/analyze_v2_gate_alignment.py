from pathlib import Path
import json
import re
import torch

ROOT = Path('/home/ubuntu/outer-loop/src/local_data/checkpoints')
INIT = ROOT / 'aklein4--forte-init' / '000000000000' / 'model.pt'
RUN = ROOT / 'aklein4--Horizon-TPU_forte-v2-1b'

def load(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'model' in state:
        state = state['model']
    return {k.replace('._orig_mod.', '.').replace('._module.', '.'): v.float() for k,v in state.items()}

def cos(x,y):
    x=x.flatten(); y=y.flatten()
    return float(torch.dot(x,y)/(x.norm()*y.norm()).clamp_min(1e-30))

def keys_for(state, needle):
    def order(k):
        group = 0 if k.startswith('backbone_layers.') else 1
        return group, int(re.search(r'\.layers\.(\d+)\.', k).group(1))
    return sorted((k for k in state if needle in k and k.endswith('.weight')), key=order)

init=load(INIT)
steps=[50,100,150,200,250,400,450,500]
states={s:load(RUN/f'{s:012d}'/'model.pt') for s in steps}
ak=keys_for(states[500], 'activation_gate_proj')
gk=keys_for(states[500], 'gradient_gate_proj')
rows=[]
for s,state in states.items():
    for layer,(a_key,g_key) in enumerate(zip(ak,gk)):
        da=state[a_key]-init[a_key]
        dg=state[g_key]-init[g_key]
        mean=(da+dg)/2
        total=da.square().sum()+dg.square().sum()
        explained=1-float(((da-mean).square().sum()+(dg-mean).square().sum())/total)
        rc=torch.nn.functional.cosine_similarity(da,dg,dim=1)
        rows.append(dict(step=s,layer=layer,delta_cos=cos(da,dg),shared_explained=explained,
                         da_norm=float(da.norm()),dg_norm=float(dg.norm()),full_weight_cos=cos(state[a_key],state[g_key]),
                         row_cos_q10=float(rc.quantile(.1)),row_cos_median=float(rc.median()),row_cos_q90=float(rc.quantile(.9))))

increments=[]
for lo,hi in zip(steps[:-1],steps[1:]):
    for layer,(a_key,g_key) in enumerate(zip(ak,gk)):
        da=states[hi][a_key]-states[lo][a_key]
        dg=states[hi][g_key]-states[lo][g_key]
        increments.append(dict(lo=lo,hi=hi,layer=layer,increment_cos=cos(da,dg),da_norm=float(da.norm()),dg_norm=float(dg.norm())))

baseline=[]
for layer,(a_key,g_key) in enumerate(zip(ak,gk)):
    da=states[500][a_key]-states[50][a_key]
    dg=states[500][g_key]-states[50][g_key]
    mean=(da+dg)/2
    total=da.square().sum()+dg.square().sum()
    rc=torch.nn.functional.cosine_similarity(da,dg,dim=1)
    baseline.append(dict(layer=layer,delta_cos=cos(da,dg),
                         shared_explained=1-float(((da-mean).square().sum()+(dg-mean).square().sum())/total),
                         da_norm=float(da.norm()),dg_norm=float(dg.norm()),row_cos_median=float(rc.median()),
                         row_cos_q10=float(rc.quantile(.1)),row_cos_q90=float(rc.quantile(.9))))

out=Path('/home/ubuntu/outer-loop/artifacts/v2_step500_gate_alignment')
out.mkdir(parents=True,exist_ok=True)
(out/'parameter_alignment.json').write_text(json.dumps({'rows':rows,'increments':increments,'step50_to_500':baseline},indent=2))
for s in steps:
    rr=[r for r in rows if r['step']==s]
    print(s, 'cos', sum(r['delta_cos'] for r in rr)/16, 'expl', sum(r['shared_explained'] for r in rr)/16,
          'range', min(r['delta_cos'] for r in rr),max(r['delta_cos'] for r in rr))
print('step500 layers')
for r in rows[-16:]: print(r)
print('increments')
for lo,hi in zip(steps[:-1],steps[1:]):
    rr=[r for r in increments if r['lo']==lo]
    print(lo,hi,sum(r['increment_cos'] for r in rr)/16,min(r['increment_cos'] for r in rr),max(r['increment_cos'] for r in rr))
print('50 to 500')
print('mean cos',sum(r['delta_cos'] for r in baseline)/16,'mean explained',sum(r['shared_explained'] for r in baseline)/16)
for r in baseline: print(r)
