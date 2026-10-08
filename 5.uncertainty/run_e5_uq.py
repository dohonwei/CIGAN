"""Reviewer 4.5: MC-dropout calibration and coverage-width analysis."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import norm

HERE=Path(__file__).resolve(); E5=HERE.parents[0]; E1ROOT=E5.parent/'1.prior_benchmark'; sys.path.insert(0,str(E1ROOT))
from e1_common import DATASETS,FRONTAL_INDICES,SOURCE_INDICES,checkpoint_path,load_dataset,load_fold,prior_path,set_deterministic_seed,transform_eeg
from e1_model import CIGANGenerator

LEVELS=np.array([.50,.60,.70,.80,.90,.95,.99])

def enable_dropout(model):
    model.eval()
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout1d): module.train()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True,choices=DATASETS); p.add_argument('--seed',type=int,default=42); p.add_argument('--passes',type=int,default=50); p.add_argument('--batch-size',type=int,default=32); a=p.parse_args()
    set_deterministic_seed(a.seed); data=load_dataset(a.dataset); fs=DATASETS[a.dataset]['fs']; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows=[]
    for fold in range(1,6):
        _,test=load_fold(a.dataset,fold); ck=torch.load(checkpoint_path(a.dataset,fold,'granger',a.seed),map_location=device,weights_only=False); prior=np.load(prior_path(a.dataset,fold,'granger',a.seed))['weights'].astype('float32')
        mean=np.asarray(ck['channel_mean']); std=np.asarray(ck['channel_std']); standardized=transform_eeg(data['eeg'][test],mean,std)[:,SOURCE_INDICES]
        model=CIGANGenerator().to(device); model.load_state_dict(ck['generator_ema']); enable_dropout(model); prior_t=torch.from_numpy(prior).to(device); draws=[]
        with torch.no_grad():
            for k in range(a.passes):
                batches=[]
                for start in range(0,len(test),a.batch_size): batches.append(model(torch.from_numpy(standardized[start:start+a.batch_size]).to(device),prior_t,fs).cpu().numpy())
                draws.append(np.concatenate(batches))
        draws=np.stack(draws); fmean=mean[FRONTAL_INDICES][None,:,None]; fstd=std[FRONTAL_INDICES][None,:,None]; draws=draws*fstd+fmean; pred_mean=draws.mean(0); pred_std=draws.std(0,ddof=1); real=data['eeg'][test][:,FRONTAL_INDICES]
        for level in LEVELS:
            z=norm.ppf((1+level)/2); lower=pred_mean-z*pred_std; upper=pred_mean+z*pred_std; covered=(real>=lower)&(real<=upper); width=upper-lower
            for subject in np.unique(data['subjects'][test]):
                mask=data['subjects'][test]==subject; rows.append({'dataset':a.dataset,'fold':fold,'subject':str(subject),'nominal_coverage':level,'picp':covered[mask].mean(),'mpiw_uv':width[mask].mean(),'rmse_uv':np.sqrt(np.mean((pred_mean[mask]-real[mask])**2)),'passes':a.passes})
    out=E5/'outputs'/a.dataset; out.mkdir(parents=True,exist_ok=True); raw=pd.DataFrame(rows); raw.to_csv(out/'uq_subject_calibration.csv',index=False); raw.groupby('nominal_coverage').agg(picp_mean=('picp','mean'),picp_std=('picp','std'),mpiw_mean_uv=('mpiw_uv','mean'),mpiw_std_uv=('mpiw_uv','std'),rmse_mean_uv=('rmse_uv','mean')).reset_index().to_csv(out/'uq_calibration_summary.csv',index=False); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
