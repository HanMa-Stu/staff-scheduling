"""Independently verify staffing constraints and optional reproduction equality."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent


def validate(root,parameters,forecast):
    p=json.loads(parameters.read_text(encoding='utf-8'))
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    days,n=int(p['days']),int(p['site']['N'])
    cap=int(p['site']['max_attendance_days'])
    checks={}
    def check(name,value):
        checks[name]=bool(value)
    check('summary_parameters',summary['parameters']==p)
    frame=pd.read_csv(root/'shift_counts.csv').sort_values(['day','shift'])
    check('shift_keys',len(frame)==days*6 and set(frame.site)=={'SC1'}
          and not frame.duplicated(['day','shift']).any()
          and set(zip(frame.day,frame['shift']))=={(j,k) for j in range(1,days+1) for k in range(1,7)})
    x1,x2,y1,y2=[frame[c].to_numpy().reshape(days,6) for c in
                ('stage1_regular','stage2_regular','stage1_temporary','stage2_temporary')]
    check('nonnegative_integer_counts',all(np.isfinite(a).all() and (a>=0).all()
          and np.equal(a,np.floor(a)).all() for a in (x1,x2,y1,y2)))
    check('equation29_counts_equal',np.array_equal(x1,x2))
    check('temporary_counts_retained',np.array_equal(y1,y2))
    check('daily_regular_limit',(x1.sum(axis=1)<=n).all())
    check('temporary_bound',(y1.sum(axis=1)<=p['temporary_big_m']).all())
    check('temporary_activation_gate',((y1.sum(axis=1)==0)|
          (x1.sum(axis=1)>=p['site']['O_threshold']*p['site']['Nij'])).all())
    hourly=pd.read_csv(forecast).y_pred_for_schedule.to_numpy()
    check('forecast_values',len(hourly)==days*24 and np.isfinite(hourly).all() and (hourly>=0).all())
    checkpoints=np.cumsum(hourly.reshape(days,24),axis=1)[:,np.array([8,13,16,20,22,24])-1]
    # Independent transcription of the corrected equation (25).
    a=np.array([[8,3,0,0,0,0],[8,8,5,1,0,0],[8,8,8,4,2,0],
                [8,8,8,8,6,4],[8,8,8,8,8,6],[8,8,8,8,8,8]])
    for stage,x,y in ((1,x1,y1),(2,x2,y2)):
        capacity=(p['m']*x+p['n']*y)@a.T
        check(f'stage{stage}_equation25',(capacity[:,:5]<=checkpoints[:,:5]+1e-7).all()
              and (capacity[:,5]>=checkpoints[:,5]-1e-7).all())
    roster=pd.read_csv(root/'employee_schedule.csv').sort_values('employee')
    check('employee_keys',len(roster)==n and set(roster.site)=={'SC1'}
          and set(roster.employee)==set(range(1,n+1)) and not roster.employee.duplicated().any())
    labels=roster[[f'day_{j}' for j in range(1,days+1)]].to_numpy()
    mapping={'OFF':0,**{f'S{k}':k for k in range(1,7)}}
    check('one_valid_label_per_employee_day',all(label in mapping for label in labels.ravel()))
    h=np.array([[mapping.get(label,-1) for label in row] for row in labels])
    observed=np.array([[np.sum(h[:,j]==k) for k in range(1,7)] for j in range(days)])
    check('equation29_individual_assignments',np.array_equal(observed,x1))
    attendance=(h>0).sum(axis=1)
    check('individual_attendance_cap',(attendance<=cap).all())
    longest=0
    for row in h:
        run=0
        for label in row:
            run=run+1 if label>0 else 0
            longest=max(longest,run)
    check('maximum_consecutive_days',longest<=p['site']['K'])
    daily=pd.read_csv(root/'daily_summary.csv').sort_values('day')
    check('daily_summary',len(daily)==days and np.array_equal(daily.day,np.arange(1,days+1))
          and np.array_equal(daily.regular_staff,x1.sum(axis=1))
          and np.array_equal(daily.temporary_staff,y1.sum(axis=1))
          and np.array_equal(daily.total_staff,(x1+y1).sum(axis=1)))
    ids=pd.read_csv(root/'day1_staff_ids.csv').sort_values('shift')
    check('first_day_ids',len(ids)==6 and ids['shift'].tolist()==list(range(1,7))
          and all(int(row.regular_count)==x1[0,int(row.shift)-1]
          and [int(s.strip()) for s in str(row.employee_ids).split(',')]==
              (np.flatnonzero(h[:,0]==int(row.shift))+1).tolist() for row in ids.itertuples()))
    selected=summary['selected_stage1']
    check('summary_totals',selected['regular_worker_days']==int(x1.sum())
          and selected['temporary_worker_days']==int(y1.sum())
          and selected['daily_regular']==x1.sum(axis=1).tolist()
          and selected['daily_temporary']==y1.sum(axis=1).tolist())
    check('forecast_hash',summary['forecast_sha256']==hashlib.sha256(forecast.read_bytes()).hexdigest())
    return dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
                regular_worker_days=int(x1.sum()),temporary_worker_days=int(y1.sum()),
                attendance_days_min=int(attendance.min()),attendance_days_max=int(attendance.max()),
                max_consecutive_days=longest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=BASE.parents[1]/'results/SC1')
    parser.add_argument('--parameters',type=Path,default=BASE/'parameters.json')
    parser.add_argument('--forecast',type=Path,default=BASE.parents[1]/'data/SC1_forecast.csv')
    parser.add_argument('--compare',type=Path,help='Optional reference result directory')
    args=parser.parse_args()
    result=validate(args.results,args.parameters,args.forecast)
    if args.compare:
        matches={name:pd.read_csv(args.results/name).equals(pd.read_csv(args.compare/name)) for name in
                 ('shift_counts.csv','daily_summary.csv','employee_schedule.csv','day1_staff_ids.csv')}
        matches['summary.json']=json.loads((args.results/'summary.json').read_text(encoding='utf-8'))==json.loads((args.compare/'summary.json').read_text(encoding='utf-8'))
        result['reproduction_matches']=matches
        if not all(matches.values()): result['status']='failed'
    (args.results/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if result['status']!='passed': raise SystemExit(1)


if __name__=='__main__':
    main()
