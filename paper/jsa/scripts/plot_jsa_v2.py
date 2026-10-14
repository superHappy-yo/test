#!/usr/bin/env python3
"""Export scientific figures from independent process-bootstrap estimates."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
s=json.loads((ROOT/'paper/data/jsa_v2/summary.json').read_text())
out=ROOT/'paper/jsa/figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})


def points(ax,records,key,offset,color,label):
 xs=[];ys=[];lower=[];upper=[]
 for i,r in enumerate(records):
  stat=r[key]
  if stat['geomean'] is None:continue
  xs.append(i+offset);ys.append(stat['geomean']);lower.append(stat['geomean']-stat['ci95'][0]);upper.append(stat['ci95'][1]-stat['geomean'])
 if xs:ax.errorbar(xs,ys,yerr=[lower,upper],fmt='o',capsize=3,color=color,label=label)

if any(r['coverage'] for r in s['selection']):
 domains=[domain for domain in ('operator','operator-vllm') if any(r['domain']==domain and r['coverage'] for r in s['selection'])]
 fig,axes=plt.subplots(len(domains),2,figsize=(7.4,2.65*len(domains)),sharex=True,squeeze=False)
 contexts=[('direct','same'),('direct','rotated'),('graph','same'),('graph','rotated')]
 for row,domain in enumerate(domains):
  for policy,offset,color in [('scic-v2',-.10,'#2166ac'),('scic-v2-no-ai',.10,'#b35806')]:
   records=[next(r for r in s['selection'] if r['domain']==domain and r['mode']==mode and r['reuse']==reuse and r['policy']==policy) for mode,reuse in contexts]
   points(axes[row,0],records,'steady_vs_full',offset,color,policy)
   points(axes[row,1],records,'total_preparation_speed',offset,color,policy)
  axes[row,0].axhline(.98,color='gray',ls='--',label='98% requirement')
  axes[row,1].axhline(1/.7,color='gray',ls='--',label='30% cost reduction')
  for col in range(2):
   ax=axes[row,col];ax.axhline(1,color='black',lw=.7);ax.grid(axis='y',alpha=.2)
   ax.set_title(('HF' if row==0 else 'vLLM')+(' steady/full calibration' if col==0 else ' total preparation speed'))
   ax.set_xticks(range(4),['Direct\nsame','Direct\nrotated','Graph\nsame','Graph\nrotated'])
   ax.set_ylabel('Speed ratio');ax.legend(fontsize=8)
 fig.tight_layout()
 for ext in ('pdf','png'):fig.savefig(out/f'fig_v2_selection.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig)

if any(r['throughput']['geomean'] is not None for r in s['serving']):
 fig,axes=plt.subplots(1,2,figsize=(7.4,3.1),sharey=True)
 for ax,domain in zip(axes,('serving','serving-pretrained')):
  records=[r for r in s['serving'] if r['domain']==domain]
  points(ax,records,'throughput',0,'#2166ac','matching full request hashes')
  ax.set_xticks(range(len(records)),[r['policy'].replace('full-calibration','Full cal.').replace('scic-v2-no-ai','No AI').replace('scic-v2','V2').replace('best-fixed','Fixed') for r in records],rotation=25,ha='right')
  ax.axhline(1,color='black',lw=.8);ax.axhline(1.02,color='gray',ls='--');ax.grid(axis='y',alpha=.2)
  ax.set_title('Random model' if domain=='serving' else 'Pretrained Qwen3-0.6B');ax.set_ylabel('Output throughput/original')
 fig.tight_layout()
 for ext in ('pdf','png'):fig.savefig(out/f'fig_v2_serving.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig)
