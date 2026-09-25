"""Recompute DBS-Gym author's evaluate_HF_DBS.py PSD on cached native LFPs.

This does not rerun six published evaluations or evaluate an RL checkpoint.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.signal import butter,filtfilt,hilbert

def author_psd(signal,dt=.0005):
    fs=1/dt
    b,a=butter(5,[12/(.5*fs),30/(.5*fs)],btype='band')
    filtered=filtfilt(b,a,signal)
    envelope=np.abs(hilbert(filtered)) # computed by upstream utility, but not used in PSD
    assert envelope.size==signal.size
    power=np.abs(np.fft.rfft(filtered)/filtered.shape[0])**2*2
    freq=np.fft.rfftfreq(filtered.shape[0],dt)
    smoothed=filtfilt(np.ones(12),5,power)
    return float(np.sum(smoothed[(freq>12.5)&(freq<21)]))

def run(off,high,out):
    waves=[];metas=[]
    for path,action in ((off,0),(high,1)):
        with np.load(path,allow_pickle=False) as z:
            assert z['current_step'].item()==z['target'].item()==5555
            assert z['seed'].item()==222 and z['action'].item()==action
            waves.append(z['wave'].copy());metas.append(hashlib.sha256(z['wave'].tobytes()).hexdigest())
    values=[author_psd(x) for x in waves]
    result={'source':'https://github.com/NevVerVer/DBS-Gym/blob/aa0b10b9502e4f62dea755ce6468da024235c319/aDBS_RL/evaluate_HF_DBS.py',
      'protocol':'Paper repository calc_psd_for_simple_eval reproduced exactly: 5th order Butterworth 12-30 Hz filtfilt; rFFT squared magnitude*2/N^2, filtfilt([1]*12,5,PSD), sum 12.5<f<21; dt .0005 s',
      'checkpoint_source':'src/parkinson_env0_chunk.py 512-neuron 5555-step Env0 full-horizon matched seed 222, actions normalized OFF=0, high=+1',
      'seed':222,'n_steps':5555,'lfp_hash_off_high':metas,'off_beta_psd':values[0], 'high_beta_psd':values[1], 'high_pct_of_off':100*values[1]/values[0],
      'published_off_pct':100,'published_high_pct':19.8,'published_high_sd_pct':1.9,
      'limitation':'Only one matched seed, not authors six-run mean/SD; author eval-env regime and initialization may differ; no learned RL controller evaluated. Not published-comparable RL result or clinical evidence.'}
    Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--off',required=True);a.add_argument('--high',required=True);a.add_argument('--out',default='results/parkinson_published_psd_seed222.json');x=a.parse_args();run(x.off,x.high,x.out)
