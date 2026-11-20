"""Reproduce SC1 staffing from saved forecasts. No forecasting model is run."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'src'))
from stage1_ga import Stage1GAConfig, GAConfig, solve_stage1_ga
from stage2_ga import FixedStage2GA


def write_csv(path,header,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters',type=Path,default=BASE/'parameters.json')
    parser.add_argument('--forecast',type=Path,default=BASE.parents[1]/'data/SC1_forecast.csv')
    parser.add_argument('--out',type=Path,default=BASE.parents[1]/'results/SC1_reproduced')
    args=parser.parse_args()
    params=json.loads(args.parameters.read_text(encoding='utf-8'))
    days,employees=int(params['days']),int(params['site']['N'])
    forecast_file=args.forecast
    with forecast_file.open(encoding='utf-8-sig',newline='') as handle:
        forecast=list(csv.DictReader(handle))
    hourly=np.array([float(row['y_pred_for_schedule']) for row in forecast])
    site={**params['site'],'demand':np.cumsum(hourly.reshape(days,24),axis=1)}
    archive={}
    best_stage1=solve_stage1_ga(site,days=days,hours=24,m=params['m'],n=params['n'],
                              temporary_big_m=params['temporary_big_m'],
                              config=Stage1GAConfig(**params['stage1_ga']),candidate_archive=archive)
    if best_stage1.x is None:
        raise RuntimeError('No Stage-1 feasible candidate')
    visited_count=len(archive)
    visited=list(archive.values())
    for candidate in visited:
        candidate['origin']='visited_by_stage1_ga'
    # Stage-1 constraints and fitness are separable by day. Retain single-day
    # crossovers of its best chromosome and its visited chromosomes. No weekly
    # attendance cap is consulted here, and no shift number is edited by hand.
    # The existing GA crossover also exchanges whole days.
    for source_candidate in visited:
        for day in range(days):
            xx,yy=best_stage1.x.copy(),best_stage1.y.copy()
            xx[day],yy[day]=source_candidate['x'][day],source_candidate['y'][day]
            key=xx.tobytes()
            if key in archive:
                continue
            balance=float(np.square(xx-xx.mean(axis=1,keepdims=True)).sum())/(employees**2)
            people=int((xx+yy).sum())
            archive[key]={'x':xx,'y':yy,'people':people,'balance':balance,
                          'fitness':people+params['stage1_ga']['balance_weight']*balance,
                          'origin':'single_day_crossover_of_stage1_ga_chromosomes',
                          'crossover_day':day+1,'source_parent_fitness':source_candidate['fitness']}
    candidates=sorted(archive.values(),key=lambda item:item['fitness'])
    history=[]
    selected=schedule=metadata=None
    for rank,candidate in enumerate(candidates,1):
        # Stage 2 rejects infeasible input; it never rescales or edits X.
        try:
            ga=FixedStage2GA(candidate['x'],candidate['y'],employees,
                            site['max_attendance_days'],site['K'],GAConfig(**params['ga']))
            schedule,metadata=ga.solve()
        except ValueError as error:
            history.append([rank,candidate['fitness'],int(candidate['x'].sum()),'rejected',str(error)])
            continue
        selected=candidate
        history.append([rank,candidate['fitness'],int(candidate['x'].sum()),'accepted','Exact employee assignment verified'])
        break
    if selected is None:
        raise RuntimeError('No assignable candidate in this Stage-1 GA archive')
    x,y=selected['x'],selected['y']
    observed=np.array([[np.sum(schedule[:,j]==k) for k in range(6)] for j in range(days)])
    assert np.array_equal(observed,x)

    args.out.mkdir(parents=True,exist_ok=True)
    shifts=['00:00-08:00','05:00-13:00','08:00-16:00','12:00-20:00','14:00-22:00','16:00-24:00']
    write_csv(args.out/'shift_counts.csv',
              ['site','day','shift','shift_time','stage1_regular','stage2_regular','stage1_temporary','stage2_temporary'],
              [['SC1',j+1,k+1,shifts[k],int(x[j,k]),int(observed[j,k]),int(y[j,k]),int(y[j,k])]
               for j in range(days) for k in range(6)])
    write_csv(args.out/'daily_summary.csv',['site','day','regular_staff','temporary_staff','total_staff'],
              [['SC1',j+1,int(x[j].sum()),int(y[j].sum()),int((x[j]+y[j]).sum())] for j in range(days)])
    write_csv(args.out/'employee_schedule.csv',['site','employee',*[f'day_{j+1}' for j in range(days)]],
              [['SC1',p+1,*[f'S{int(k)+1}' if k>=0 else 'OFF' for k in schedule[p]]] for p in range(employees)])
    write_csv(args.out/'day1_staff_ids.csv',['site','day','shift','regular_count','employee_ids'],
              [['SC1',1,k+1,int(x[0,k]),', '.join(str(p+1) for p in np.flatnonzero(schedule[:,0]==k))] for k in range(6)])
    summary={
        'status':'feasible','site':'SC1','parameters':params,
        'forecast_sha256':hashlib.sha256(forecast_file.read_bytes()).hexdigest(),
        'forecast_retrained':False,'stage1_constraints_changed':False,
        'stage1_fitness_changed':False,'equation29_exact':True,
        'selected_stage1':{'regular_worker_days':int(x.sum()),'temporary_worker_days':int(y.sum()),
             'total_worker_days':int((x+y).sum()),'daily_regular':x.sum(axis=1).tolist(),
             'daily_temporary':y.sum(axis=1).tolist(),'first_day_regular':x[0].tolist(),
             'fitness':selected['fitness'],'regular_shift_balance':selected['balance']},
        'stage2_ga':metadata,
        'search':{'stage1_seed':params['stage1_ga']['seed'],
             'stage1_generations_completed':best_stage1.metadata['generations_completed'],
             'visited_candidates':visited_count,'candidates_including_day_crossovers':len(candidates),
             'selected_fitness_rank':rank,'selected_origin':selected['origin'],
             'crossover_day':selected.get('crossover_day'),
             'source_parent_fitness':selected.get('source_parent_fitness'),
             'unselected_stage1_best_regular_worker_days':int(best_stage1.x.sum()),
             'rule':'Rank Stage-1 candidates and single-day crossovers by original fitness; Stage 2 rejects unassignable inputs without changing counts.'},
        'scope':'Feasible heuristic solution for SC1; no claim of global optimality.'}
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':summary['status'],**summary['selected_stage1']},indent=2))


if __name__=='__main__':
    main()
