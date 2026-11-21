"""Plot saved hourly SARIMA-LSTM forecasts without training any model.

Run from the repository root: python code/forecast/plot_hourly_forecasts.py
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(os.environ.get('TEMP','/tmp'))/'hourly_forecast_mplconfig'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'results/forecast/hourly'
OUT=RESULT/'figures'
OUT.mkdir(parents=True,exist_ok=True)
STATIONS=OUT/'stations';STATIONS.mkdir(exist_ok=True)
pred=pd.read_csv(RESULT/'SARIMA_LSTM_predictions.csv.gz')
assert len(pred)==36765 and pred.model_name.eq('SARIMA-LSTM').all()
sites=sorted(pred.site.unique(),key=lambda s:int(s[2:]))
assert len(sites)==57 and set(pred.seed)=={42,43,44}
mean=pred.groupby(['site','source_row_1based'],as_index=False).agg(y_true=('y_true','first'),y_pred_mean=('y_pred','mean'),seeds=('seed','size'))
assert mean.seeds.eq(3).all()
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.linewidth':.7,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
def draw(ax,site,label=None):
    raw=pd.read_csv(ROOT/'data/hourly'/f'{site}.csv',header=None).iloc[:,0].to_numpy(float)
    d=mean[mean.site==site].sort_values('source_row_1based')
    assert len(raw)==720 and len(d)==215
    np.testing.assert_array_equal(d.y_true.to_numpy(),raw[505:])
    ax.plot(np.arange(1,721),raw,color='blue',lw=.65,label='Observed')
    ax.plot(d.source_row_1based,d.y_pred_mean,color='red',lw=.65,marker='.',markersize=1.5,label='Forecast')
    ax.axvline(505.5,color='.6',ls='--',lw=.5,zorder=0)
    ax.set(xlabel='Time (hour)',ylabel='Cargo (pieces)',xlim=(1,720),title=site)
    ax.set_xticks(np.arange(0,721,100));ax.set_xlim(1,720)
    f=ScalarFormatter(useMathText=True);f.set_powerlimits((-3,4));f.set_useOffset(False);ax.yaxis.set_major_formatter(f)
    ax.tick_params(length=3,width=.65,labelsize=9)
    ax.legend(loc='best',fontsize=8,framealpha=1,edgecolor='black',fancybox=False,borderpad=.3,handlelength=2)
    if label:ax.text(.5,-.25,label,transform=ax.transAxes,ha='center',fontsize=10)

for site in sites:
    fig,ax=plt.subplots(figsize=(8.0,3.6));draw(ax,site);fig.tight_layout(pad=.8)
    fig.savefig(STATIONS/f'Hourly_SARIMA_LSTM_{site}.png',dpi=400);plt.close(fig)
with PdfPages(OUT/'Hourly_SARIMA_LSTM_all_stations.pdf') as pdf:
    for start in range(0,len(sites),2):
        fig,axes=plt.subplots(2,1,figsize=(7.2,7.8))
        for i,ax in enumerate(axes):
            if start+i<len(sites):draw(ax,sites[start+i],f'({chr(97+i)})')
            else:ax.axis('off')
        fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.95,hspace=.50)
        fig.text(.10,.025,'Hourly SARIMA-LSTM fixed-origin forecasts. Red: mean of seeds 42, 43, 44.\n505 training hours; forecasts cover hours 506-720 (215 hours).',fontsize=8)
        pdf.savefig(fig)
        if start==0:
            fig.savefig(OUT/'Hourly_SARIMA_LSTM_SC1_SC2.png',dpi=400)
            fig.savefig(OUT/'Hourly_SARIMA_LSTM_SC1_SC2.pdf')
        plt.close(fig)
print('Saved57 hourly station figures and PDF atlas:',OUT)
