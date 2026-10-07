"""Fixed cleaning rules shared by training and inference."""
import numpy as np
import pandas as pd
from .config import FEATURES, DISTANCE_BINS, DISTANCE_LABELS


def clean_data(data):
    result = data.copy()
    result['date'] = pd.to_datetime(result['date'])
    result['weight_missing'] = result.weight.isna().astype('int8')
    result['weight_was_negative'] = result.weight.lt(0).astype('int8')
    result['weight'] = result.weight.abs()
    result['market_index_missing'] = result.market_index.isna().astype('int8')
    if not np.isfinite(result.distance).all() or not result.distance.gt(0).all():
        raise ValueError('Distance must be finite and positive')
    if not result.equipment.isin(['Dry Van', 'Flatbed', 'Reefer']).all():
        raise ValueError('Unrecognized equipment')
    return result


def haversine(data):
    a, b, c, d = [np.radians(data[col]) for col in
                   ['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']]
    value = np.sin((c-a)/2)**2 + np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 3958.8 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def build_features(data, columns=None):
    data = clean_data(data)
    columns = FEATURES if columns is None else columns
    result = data[['distance', 'weight', 'weight_missing', 'weight_was_negative',
                   'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
                   'market_index', 'market_index_missing']].copy()
    for name in ['Dry Van', 'Flatbed', 'Reefer']:
        result['eq_' + name.replace(' ', '')] = data.equipment.eq(name).astype('int8')
    if set(columns).intersection({'quote_signal', 'day_qs_median', 'day_low_qs_share'}):
        result['quote_signal'] = data.quote_signal
        result['day_qs_median'] = data.groupby('date').quote_signal.transform('median')
        result['day_low_qs_share'] = data.quote_signal.lt(1.7).groupby(data.date).transform('mean')
    if 'circuity' in columns:
        result['circuity'] = (data.distance / haversine(data).replace(0, np.nan)).clip(upper=3)
    if 'weekday' in columns:
        result['weekday'] = data.date.dt.dayofweek
    if 'log_distance' in columns:
        result['log_distance'] = np.log(data.distance)
    result = result.loc[:, columns]
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError('Infinite feature values')
    return result


def distance_groups(data):
    return pd.cut(data.distance, DISTANCE_BINS, labels=DISTANCE_LABELS)


def daily_behaviour(data):
    rows = []
    for date, part in data.groupby('date'):
        correlation = part.quote_signal.corr(np.log(part.distance))
        label = 'follows price' if correlation < -0.5 else 'upside down' if correlation > 0.5 else 'flat'
        rows.append(dict(date=date, correlation=correlation, behaviour=label))
    return pd.DataFrame(rows)
