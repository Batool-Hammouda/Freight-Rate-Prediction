"""Development-only comparisons and a guarded final holdout evaluation."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .config import ROOT, PARAMS, FEATURES, CORE_FEATURES, QUOTE_FEATURES, SEED, CUTOFF
from .features import distance_groups
from .train import cross_validate, fit_model, predict_dollars, metrics
from .run import read_training

EVIDENCE=ROOT/'reports/evidence'


def compare_models(development):
    specifications=[('Quote reference',None,'quote'),('Median RPM',None,'median'),
                    ('LightGBM with quote',CORE_FEATURES+QUOTE_FEATURES,None),
                    ('LightGBM core',CORE_FEATURES,None),('LightGBM final',FEATURES,None)]
    tables=[]
    for name,columns,baseline in specifications:
        print('Comparison:',name,flush=True)
        table=cross_validate(development,columns=columns,baseline=baseline)
        table.insert(0,'model',name); tables.append(table)
    results=pd.concat(tables,ignore_index=True)
    results.to_csv(EVIDENCE/'cv_folds.csv',index=False,float_format='%.6f')
    means=results.groupby('model',sort=False)[['MAE','MAPE','RMSE','R2']].mean().round(2)
    means.to_csv(EVIDENCE/'cv_summary.csv')
    print(means.to_string(),flush=True)
    return results


def tune(development):
    # A second fixed seed checks whether a small gain survives sampling variation.
    seeds=[SEED,SEED+1]
    trials=[('default',{},None),('alpha_0.5',{'alpha':.5},None),
            ('alpha_2',{'alpha':2.0},None),('leaves_15',{'num_leaves':15},None),
            ('min_child_80',{'min_child_samples':80},None),
            ('circuity',{},'circuity'),('weekday',{},'weekday'),('log_distance',{},'log_distance')]
    rows=[]; details=[]
    for name,changes,extra in trials:
        cols=FEATURES+([extra] if extra else [])
        folds=[]
        for seed in seeds:
            result=cross_validate(development,params={**changes,'random_state':seed},columns=cols)
            result['trial']=name;result['seed']=seed;details.append(result)
            folds.append(result)
        all_folds=pd.concat(folds)
        row=dict(trial=name,changes=json.dumps(changes,sort_keys=True),extra=extra or '',
                 mean_MAE=all_folds.MAE.mean(),august_MAE=all_folds.loc[all_folds.month.eq('2025-08'),'MAE'].mean())
        rows.append(row);print('Trial',name,round(row['mean_MAE'],2),round(row['august_MAE'],2),flush=True)
    table=pd.DataFrame(rows)
    baseline=table.iloc[0]
    table['improvement_pct']=100*(1-table.mean_MAE/baseline.mean_MAE)
    table['eligible']=(table.improvement_pct>1)&((table.extra=='')|(table.august_MAE<=baseline.august_MAE))
    # Select a tested configuration rather than combining changes that were never tested.
    eligible=table.loc[table.eligible]
    selected=eligible.sort_values('mean_MAE').iloc[0] if len(eligible) else baseline
    table['selected']=table.trial.eq(selected.trial)
    table.round(4).to_csv(EVIDENCE/'tuning.csv',index=False)
    pd.concat(details).to_csv(EVIDENCE/'tuning_folds.csv',index=False,float_format='%.6f')
    choice={'params':{**PARAMS,**json.loads(selected.changes)},
            'features':FEATURES+([selected.extra] if selected.extra else []),
            'selected_trial':selected.trial,'seeds':seeds,
            'policy':'Mean MAE improvement greater than 1%; extra features must not worsen August. No untested combinations.',
            'locked_at_utc':datetime.now(timezone.utc).isoformat()}
    (ROOT/'models/selection.json').write_text(json.dumps(choice,indent=2))
    print('Locked choice:',choice,flush=True)
    chosen=cross_validate(development,params=choice['params'],columns=choice['features'])
    chosen.round(6).to_csv(EVIDENCE/'selected_cv_folds.csv',index=False)
    return choice


def unseen_city_check(development,choice):
    training=development.loc[development.date<'2025-08-01']
    august=development.loc[development.date>='2025-08-01']
    cities=sorted(set(training.pickup)|set(training.delivery))
    hidden=sorted(np.random.default_rng(SEED).choice(cities,size=8,replace=False).tolist())
    keep=~training.pickup.isin(hidden)&~training.delivery.isin(hidden)
    affected=august.pickup.isin(hidden)|august.delivery.isin(hidden)
    rows=[]
    for label,subset in [('Cities seen in training',training),('Cities hidden from training',training.loc[keep])]:
        model=fit_model(subset,choice['params'],choice['features'])
        predictions,_=predict_dollars(model,august.loc[affected],choice['features'])
        rows.append(dict(condition=label,train_rows=len(subset),evaluation_rows=int(affected.sum()),
                         **metrics(august.loc[affected,'posted_rate'],predictions)))
    table=pd.DataFrame(rows).round(2)
    table.to_csv(EVIDENCE/'unseen_city_test.csv',index=False)
    (EVIDENCE/'hidden_cities.json').write_text(json.dumps(hidden,indent=2))
    print('Hidden cities:',hidden);print(table.to_string(index=False),flush=True)


def score_holdout_once(train,choice):
    target=EVIDENCE/'holdout_metrics.csv'
    choice_hash=hashlib.sha256(json.dumps({'params':choice['params'],'features':choice['features']},sort_keys=True).encode()).hexdigest()
    ledger_path=EVIDENCE/'holdout_ledger.json'
    if ledger_path.exists():
        ledger=json.loads(ledger_path.read_text())
        if ledger['choice_sha256']!=choice_hash:raise RuntimeError('Holdout already used with different choices')
        if not target.exists():raise RuntimeError('Holdout run was interrupted; inspect ledger without silently rescoring')
        print('Existing final holdout results retained; no rescore.',flush=True)
        return
    ledger={'choice_sha256':choice_hash,'started_at_utc':datetime.now(timezone.utc).isoformat(),
            'new_pipeline_evaluation_count':1,
            'prior_exposure':'Earlier Ridge work in this project already evaluated September-October. This is not a historically untouched holdout. The current LightGBM choices use only January-August folds.'}
    ledger_path.write_text(json.dumps(ledger,indent=2))
    development=train.loc[train.date<CUTOFF];holdout=train.loc[train.date>=CUTOFF]
    model=fit_model(development,choice['params'],choice['features'])
    predicted,_=predict_dollars(model,holdout,choice['features'])
    predictions={'Selected LightGBM':predicted,'Quote reference':holdout.quote_signal*holdout.distance,
                 'Median RPM':(development.posted_rate/development.distance).median()*holdout.distance}
    rows=[]
    for name,values in predictions.items():
        series=pd.Series(np.asarray(values),index=holdout.index)
        for grouping,groups in [('overall',pd.Series('all',index=holdout.index)),
                                ('month',holdout.date.dt.strftime('%Y-%m')),
                                ('equipment',holdout.equipment),('distance',distance_groups(holdout))]:
            for group,indices in groups.groupby(groups,observed=True).groups.items():
                rows.append(dict(model=name,grouping=grouping,group=str(group),rows=len(indices),
                                 **metrics(holdout.loc[indices,'posted_rate'],series.loc[indices])))
    table=pd.DataFrame(rows).round(2);table.to_csv(target,index=False)
    ledger['completed_at_utc']=datetime.now(timezone.utc).isoformat();ledger_path.write_text(json.dumps(ledger,indent=2))
    print(table.loc[table.grouping.isin(['overall','month'])].to_string(index=False),flush=True)


def main(skip_tuning=False):
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    if (EVIDENCE/'holdout_ledger.json').exists():
        raise RuntimeError('Evaluation already completed. Read stored results; no further selection or rescore.')
    train=read_training();development=train.loc[train.date<CUTOFF]
    compare_models(development)
    if skip_tuning:
        choice={'params':PARAMS,'features':FEATURES,'selected_trial':'default','seeds':[SEED]}
        (ROOT/'models/selection.json').write_text(json.dumps(choice,indent=2))
        cross_validate(development).to_csv(EVIDENCE/'selected_cv_folds.csv',index=False)
    else:choice=tune(development)
    unseen_city_check(development,choice)
    score_holdout_once(train,choice)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--skip-tuning',action='store_true')
    main(parser.parse_args().skip_tuning)
