"""Reviewer 6.1/6.5: controlled synthetic EEG-like signals with known truth."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.signal import chirp
ROOT=Path(__file__).resolve().parents[0]

def pink(rng,n):
    f=np.fft.rfftfreq(n); scale=np.zeros_like(f); scale[1:]=1/np.sqrt(f[1:]); x=np.fft.irfft((rng.normal(size=len(f))+1j*rng.normal(size=len(f)))*scale,n); return x/(x.std()+1e-12)
def clean(kind,t,rng):
    if kind=='multisine': return sum(a*np.sin(2*np.pi*f*t+rng.uniform(0,2*np.pi)) for a,f in [(8,6),(5,10),(3,20),(2,38)])
    if kind=='chirp': return 8*chirp(t,f0=4,f1=42,t1=t[-1],method='linear')+3*np.sin(2*np.pi*10*t)
    centers=rng.uniform(.5,t[-1]-.5,5); return sum(rng.choice([-1,1])*12*np.exp(-.5*((t-c)/.08)**2) for c in centers)+2*np.sin(2*np.pi*8*t)
def add_noise(x,kind,snr,rng):
    if kind=='white': n=rng.normal(size=x.shape)
    elif kind=='pink': n=np.stack([pink(rng,x.shape[-1]) for _ in range(x.shape[0])])
    else:
        n=np.zeros_like(x); count=max(1,x.shape[-1]//100); idx=rng.integers(0,x.shape[-1],size=(x.shape[0],count));
        for c in range(x.shape[0]): n[c,idx[c]]=rng.normal(0,10,size=count)
    n=n/(n.std(axis=-1,keepdims=True)+1e-12); power=np.mean(x*x,axis=-1,keepdims=True); return x+n*np.sqrt(power/(10**(snr/10)))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--fs',type=int,default=128); p.add_argument('--seconds',type=int,default=30); p.add_argument('--repeats',type=int,default=20); p.add_argument('--seed',type=int,default=42); a=p.parse_args(); rng=np.random.default_rng(a.seed); t=np.arange(a.fs*a.seconds)/a.fs; rows=[]; clean_bank=[]; target_bank=[]
    for signal in ['multisine','chirp','erp']:
      for rep in range(a.repeats):
        sources=np.stack([clean(signal,t,rng)+rng.normal(0,.3,len(t)) for _ in range(30)]).astype('float32'); targets=np.stack([.45*sources[0]+.30*sources[4]+.25*sources[12],.40*sources[1]+.35*sources[5]+.25*sources[18]]).astype('float32'); source_id=len(clean_bank); clean_bank.append(sources); target_bank.append(targets)
        for noise in ['white','pink','impulsive']:
          for snr in [20,10,5,0,-5]: rows.append((signal,rep,noise,snr,source_id,add_noise(sources,noise,snr,rng).astype('float32')))
    out=ROOT/'data'; out.mkdir(parents=True,exist_ok=True); np.savez(out/'synthetic_noise_benchmark.npz',signal_type=np.array([r[0] for r in rows]),repeat=np.array([r[1] for r in rows]),noise_type=np.array([r[2] for r in rows]),snr_db=np.array([r[3] for r in rows]),source_id=np.array([r[4] for r in rows]),noisy_sources=np.stack([r[5] for r in rows]),clean_source_bank=np.stack(clean_bank),true_target_bank=np.stack(target_bank),fs=a.fs); (out/'manifest.json').write_text(json.dumps(vars(a),indent=2),encoding='utf-8'); print(out)
if __name__=='__main__': main()
