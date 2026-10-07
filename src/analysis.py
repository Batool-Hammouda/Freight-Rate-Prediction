"""Evidence tables used directly by the notebook and report."""
import json
import numpy as np
import pandas as pd
from .config import ROOT, CUTOFF
from .features import clean_data, daily_behaviour, distance_groups, haversine
from .run import read_training

E=ROOT/'reports/evidence'

def save_table(name,table):
    E.mkdir(parents=True,exist_ok=True)
    table.to_csv(E/(name+'.csv'),index=True,float_format='%.2f')
    print(name);print(table.round(2).to_string());print()
    return table

def dataset_summary(data):
    return dict(rows=len(data),columns=len(data.columns),start=str(data.date.min().date()),end=str(data.date.max().date()),
                duplicate_rows=int(data.duplicated().sum()),duplicate_ids=int(data.load_id.duplicated().sum()),
                missing_weight=int(data.weight.isna().sum()),negative_weight=int(data.weight.lt(0).sum()),
                missing_market_index=int(data.market_index.isna().sum()),nonpositive_distance=int(data.distance.le(0).sum()))

def quote_tables(data,prefix):
    daily=daily_behaviour(data)
    daily.to_csv(E/(prefix+'_daily.csv'),index=False,float_format='%.6f')
    monthly=pd.crosstab(daily.date.dt.strftime('%Y-%m'),daily.behaviour)
    save_table(prefix+'_behaviour',monthly)
    groups=distance_groups(data)
    quotes=data.pivot_table(index=data.date.dt.strftime('%Y-%m'),columns=groups,values='quote_signal',aggfunc='median',observed=True)
    save_table(prefix+'_quote_medians',quotes)
    corr=daily.groupby('behaviour').correlation.agg(['min','max','count']).round(2)
    save_table(prefix+'_correlation_ranges',corr)
    if 'posted_rate' in data:
        rpm=data.assign(rpm=data.posted_rate/data.distance).pivot_table(index=data.date.dt.strftime('%Y-%m'),columns=groups,values='rpm',aggfunc='median',observed=True)
        save_table(prefix+'_rpm_medians',rpm)
    return daily

def explore_training():
    data=read_training()
    summary=dataset_summary(data)
    summary['price_distance_correlation']=round(float(data.posted_rate.corr(data.distance)),2)
    rpm=data.posted_rate/data.distance
    summary['rpm_above_8']=int(rpm.gt(8).sum());summary['rpm_max']=round(float(rpm.max()),2)
    summary['rpm_median']=round(float(rpm.median()),2)
    summary['distance_min']=round(float(data.distance.min()),2);summary['distance_max']=round(float(data.distance.max()),2)
    (E/'training_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    save_table('quality',data.isna().sum().rename('missing').to_frame())
    save_table('training_description',data.select_dtypes('number').describe().T)
    save_table('weight_comparison',pd.DataFrame({'negative_absolute':data.loc[data.weight.lt(0),'weight'].abs().describe(),
                                               'positive':data.loc[data.weight.gt(0),'weight'].describe()}).T)
    weight_prices=data.assign(negative=data.weight.lt(0),rpm=rpm).groupby('negative').rpm.agg(['count','median'])
    save_table('weight_prices',weight_prices)
    equipment=data.groupby('equipment').size().rename('rows').to_frame()
    equipment['negative']=data.weight.lt(0).groupby(data.equipment).sum()
    equipment['negative_pct']=100*equipment.negative/equipment.rows
    save_table('weight_equipment',equipment)
    save_table('training_rpm_groups',data.assign(rpm=rpm).groupby(distance_groups(data),observed=True).rpm.agg(['count','median']))
    circuity=data.distance/haversine(data)
    save_table('circuity',circuity.quantile([.001,.01,.5,.99,.999]).rename('road_to_straight_ratio').to_frame())
    daily=quote_tables(data,'training')
    cleaned=clean_data(data)
    cleaned_summary={'rows_before':len(data),'rows_after':len(cleaned),'negative_after':int(cleaned.weight.lt(0).sum()),
                     'missing_weight_after':int(cleaned.weight.isna().sum()),'missing_market_after':int(cleaned.market_index.isna().sum())}
    (E/'cleaning_summary.json').write_text(json.dumps(cleaned_summary,indent=2));print(cleaned_summary)
    return data

def check_external(training):
    data=pd.read_csv(ROOT/'data/validation.csv',parse_dates=['date'])
    template=pd.read_csv(ROOT/'data/validation_predictions_template.csv')
    summary=dataset_summary(data)
    summary['same_schema_without_target']=data.columns.tolist()==[c for c in training.columns if c!='posted_rate']
    summary['template_same_ids']=set(template.load_id)==set(data.load_id)
    summary['template_same_order']=template.load_id.tolist()==data.load_id.tolist()
    cities=set(training.pickup)|set(training.delivery)
    unseen=sorted((set(data.pickup)|set(data.delivery))-cities)
    affected=data.pickup.isin(unseen)|data.delivery.isin(unseen)
    summary.update(unseen_cities=unseen,unseen_city_count=len(unseen),affected_loads=int(affected.sum()),affected_pct=round(float(100*affected.mean()),2))
    daily=quote_tables(data,'external')
    summary['flat_days']=int(daily.behaviour.eq('flat').sum());summary['days']=len(daily)
    (E/'external_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    save_table('external_missing',data.isna().sum().rename('missing').to_frame())
    save_table('external_equipment',data.equipment.value_counts().rename('rows').to_frame())
    assert summary['same_schema_without_target'] and summary['template_same_ids'] and summary['template_same_order']
    assert data.load_id.is_unique and data.distance.gt(0).all()
    return data

def split_summary(data):
    table=pd.DataFrame([{'split':name,'rows':len(part),'start':str(part.date.min().date()),'end':str(part.date.max().date())}
                       for name,part in [('Development',data.loc[data.date<CUTOFF]),('Test',data.loc[data.date>=CUTOFF])]]).set_index('split')
    return save_table('split',table)

if __name__=='__main__':
    training=explore_training();external=check_external(training);split_summary(training)
