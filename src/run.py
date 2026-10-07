"""Refit the chosen model, create both prediction files, and run the supplied scorer."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
from .config import ROOT, FEATURES, PARAMS
from .features import build_features, distance_groups
from .train import fit_model, predict_dollars


def read_training():
    return pd.read_csv(ROOT/'data/train_test.csv', parse_dates=['date']).sort_values(['date', 'load_id']).reset_index(drop=True)


def city_coordinates(train):
    points = pd.concat([train[['pickup','pickup_lat','pickup_lon']].set_axis(['city','lat','lon'],axis=1),
                        train[['delivery','delivery_lat','delivery_lon']].set_axis(['city','lat','lon'],axis=1)])
    counts = points.groupby('city')[['lat','lon']].nunique()
    if not counts.eq(1).all().all():
        raise ValueError('City coordinates are inconsistent')
    return points.drop_duplicates().set_index('city')


def run_final(use_default=False):
    for folder in ['models','reports/evidence','scorer_results']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    evidence = ROOT/'reports/evidence'
    train = read_training()
    external = pd.read_csv(ROOT/'data/validation.csv', parse_dates=['date'])
    template = pd.read_csv(ROOT/'data/validation_predictions_template.csv')
    choice_path = ROOT/'models/selection.json'
    choice = json.loads(choice_path.read_text()) if choice_path.exists() and not use_default else {'params':PARAMS,'features':FEATURES}
    columns, params = choice['features'], choice['params']
    assert not any('qs' in c or 'quote' in c for c in columns)
    model = fit_model(train, params, columns)
    joblib.dump(model, ROOT/'models/final_model.joblib')
    (ROOT/'models/features.json').write_text(json.dumps(columns,indent=2))
    (ROOT/'models/final_settings.json').write_text(json.dumps(params,indent=2))
    predictions, floors = predict_dollars(model, external, columns)
    assert external.load_id.is_unique and template.load_id.is_unique
    assert set(template.load_id) == set(external.load_id)
    submission = template[['load_id']].merge(pd.DataFrame({'load_id':external.load_id,'predicted_rate':predictions}),
                                            on='load_id', how='left', validate='one_to_one', sort=False)
    assert submission.load_id.tolist() == template.load_id.tolist()
    assert submission.predicted_rate.notna().all() and submission.predicted_rate.gt(0).all()
    submission.to_csv(ROOT/'validation_predictions.csv',index=False)
    december_path = ROOT/'data/december_chart_inputs.csv'
    original_path = ROOT/'data/december_chart_inputs.original.csv'
    if not original_path.exists():
        shutil.copy2(december_path,original_path)
    original = pd.read_csv(original_path)
    december = original.copy()
    dates = pd.to_datetime(december.date)
    coordinates = city_coordinates(train)
    augmented = december.copy()
    augmented['date'] = dates
    for prefix,city in [('pickup','pickup'),('delivery','delivery')]:
        augmented[prefix+'_lat'] = december[city].map(coordinates.lat)
        augmented[prefix+'_lon'] = december[city].map(coordinates.lon)
    daily_market = external.groupby('date').market_index.median()
    augmented['market_index'] = dates.map(daily_market)
    assert augmented[['pickup_lat','pickup_lon','delivery_lat','delivery_lon','market_index']].notna().all().all()
    december['predicted_rate'], december_floors = predict_dollars(model, augmented, columns)
    pd.testing.assert_frame_equal(december.drop(columns='predicted_rate'), original.drop(columns='predicted_rate'))
    december.to_csv(december_path,index=False)
    augmented['predicted_rate'] = december.predicted_rate
    augmented.to_csv(evidence/'december_augmented.csv',index=False)
    result = subprocess.run([sys.executable,str(ROOT/'score.py'),'--predictions','validation_predictions.csv',
                             '--december-predictions','data/december_chart_inputs.csv'],cwd=ROOT,capture_output=True,text=True)
    (evidence/'scorer_log.txt').write_text(result.stdout+result.stderr)
    print(result.stdout,flush=True)
    if result.returncode:raise RuntimeError(result.stderr)
    imp = pd.DataFrame({'feature':columns,'gain':model.booster_.feature_importance('gain'),
                        'splits':model.booster_.feature_importance('split')}).sort_values('gain',ascending=False)
    imp['gain_pct']=100*imp.gain/imp.gain.sum()
    imp.round(2).to_csv(evidence/'feature_importance.csv',index=False)
    import matplotlib.pyplot as plt
    ax=imp.sort_values('gain').plot.barh(x='feature',y='gain_pct',legend=False,figsize=(8,5))
    ax.set(title='Final model feature importance',xlabel='Share of training gain (%)',ylabel='Feature')
    plt.tight_layout();plt.savefig(evidence/'feature_importance.png',dpi=160);plt.close()
    cities=set(train.pickup)|set(train.delivery)
    unseen=~external.pickup.isin(cities)|~external.delivery.isin(cities)
    sanity=external[['distance']].assign(prediction=predictions,rpm=predictions/external.distance,unseen_city=unseen)
    sanity.groupby('unseen_city').agg(rows=('prediction','size'),minimum=('prediction','min'),median=('prediction','median'),maximum=('prediction','max')).round(2).to_csv(evidence/'unseen_predictions.csv')
    actual=train.assign(rpm=train.posted_rate/train.distance).groupby(distance_groups(train),observed=True).rpm.median()
    predicted=sanity.groupby(distance_groups(external),observed=True).rpm.median()
    pd.DataFrame({'training_median_rpm':actual,'prediction_median_rpm':predicted}).round(2).to_csv(evidence/'rpm_sanity.csv')
    comparable=train.loc[train.equipment.eq('Dry Van') & train.distance.between(300,420),'posted_rate']
    by_month=train.loc[train.equipment.eq('Dry Van') & train.distance.between(300,420)].groupby(train.date.dt.to_period('M')).posted_rate.median()
    by_month.round(2).to_csv(evidence/'comparable_monthly_medians.csv',header=['median_dollars'])
    summary=dict(training_rows=len(train),prediction_rows=len(submission),prediction_missing=int(submission.isna().sum().sum()),
                 prediction_min=round(float(predictions.min()),2),prediction_max=round(float(predictions.max()),2),floors=floors,
                 december_rows=len(december),december_min=round(float(december.predicted_rate.min()),2),december_max=round(float(december.predicted_rate.max()),2),december_floors=december_floors,
                 comparable_rows=len(comparable),comparable_median=round(float(comparable.median()),2),comparable_q25=round(float(comparable.quantile(.25)),2),comparable_q75=round(float(comparable.quantile(.75)),2),
                 comparable_monthly_median_min=round(float(by_month.min()),2),comparable_monthly_median_max=round(float(by_month.max()),2),
                 varying_december_features=build_features(augmented,columns).nunique().loc[lambda x:x>1].index.tolist(),
                 scorer_sha256=hashlib.sha256((ROOT/'score.py').read_bytes()).hexdigest())
    (evidence/'prediction_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--default',action='store_true')
    run_final(parser.parse_args().default)
