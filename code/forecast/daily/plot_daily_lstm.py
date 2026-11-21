"""Plot saved daily LSTM forecasts without fitting any model. Run from any working directory. Requires matplotlib; outputs figures under the daily results directory."""
from pathlib import Path
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[3]
OUT=BASE/'results/forecast/daily/figures'
OUT.mkdir(parents=True,exist_ok=True)
CODE=BASE
predict=pd.read_csv(BASE/'results/forecast/daily/daily_predictions.csv')
sites=sorted(predict.site.unique(),key=lambda s:int(s[2:]))
best=predict[predict.model=='LSTM']
plotdata=best.pivot(index=['site','day','y_true'],columns='seed',values='y_pred').reset_index().rename(columns={42:'seed42',43:'seed43',44:'seed44'})
plotdata['prediction_mean']=plotdata[['seed42','seed43','seed44']].mean(axis=1)
plotdata.to_csv(OUT/'Daily_LSTM_predictions.csv',index=False)
import os
os.environ['MPLCONFIGDIR']=str(Path(os.environ.get('TEMP', '/tmp'))/'daily_lstm_mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter,MaxNLocator
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.linewidth':.7,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
pngdir=OUT/'stations';pngdir.mkdir(exist_ok=True)
def panel(ax,site,label=None):
    raw=pd.read_csv(CODE/'data/daily'/f'{site}.csv',header=None).iloc[:,0].to_numpy(float)
    d=plotdata[plotdata.site==site].sort_values('day')
    ax.plot(np.arange(1,123),raw,color='blue',lw=.85,label='Observed')
    ax.plot(d.day,d.prediction_mean,color='red',lw=.85,marker='.',markersize=2.5,label='Forecast')
    ax.axvline(86.5,color='.6',lw=.55,ls='--',zorder=0)
    ax.set(xlabel='Time (day)',ylabel='Cargo (pieces)',xlim=(1,122))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7,integer=True))
    fmt=ScalarFormatter(useMathText=True);fmt.set_powerlimits((-3,4));fmt.set_useOffset(False);ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(length=3,width=.65,labelsize=9)
    ax.set_title(site,fontsize=10,pad=6)
    ax.legend(loc='best',fontsize=8,framealpha=1,edgecolor='black',fancybox=False,borderpad=.3,handlelength=2)
    if label:ax.text(.5,-.26,label,transform=ax.transAxes,ha='center',fontsize=10)
for site in sites:
    fig,ax=plt.subplots(figsize=(5.1,3.65));panel(ax,site);fig.tight_layout(pad=.85)
    fig.savefig(pngdir/f'Daily_LSTM_{site}.png',dpi=400);plt.close(fig)
with PdfPages(OUT/'Daily_LSTM_all_stations.pdf') as pdf:
    for start in range(0,len(sites),4):
        group=sites[start:start+4]
        fig,axes=plt.subplots(2,2,figsize=(7.2,7.15))
        # Original paper places (a),(b) down the left column, (c),(d) down the right.
        axes=[axes[0,0],axes[1,0],axes[0,1],axes[1,1]]
        for i,ax in enumerate(axes):
            if i<len(group):panel(ax,group[i],f'({chr(97+i)})')
            else:ax.axis('off')
        fig.subplots_adjust(left=.10,right=.98,bottom=.18,top=.95,hspace=.55,wspace=.30)
        fig.text(.10,.025,'Daily one-step LSTM forecasts. Red: mean of seeds 42, 43, 44.\nDashed boundary: 86 training days; forecasts cover days 87-122.',fontsize=8)
        pdf.savefig(fig)
        if start==0:
            fig.savefig(OUT/'Daily_LSTM_SC1_SC4.png',dpi=400)
            fig.savefig(OUT/'Daily_LSTM_SC1_SC4.pdf')
        plt.close(fig)
