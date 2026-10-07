import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from .config import PARAMS, FEATURES, FOLD_MONTHS
from .features import build_features


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    error = actual - predicted
    return dict(MAE=float(np.abs(error).mean()), MAPE=float(100*np.mean(np.abs(error)/actual)),
                RMSE=float(np.sqrt(np.mean(error**2))),
                R2=float(1-np.sum(error**2)/np.sum((actual-actual.mean())**2)))


def fit_model(data, params=None, columns=None):
    settings = {**PARAMS, **(params or {})}
    columns = FEATURES if columns is None else columns
    return LGBMRegressor(**settings).fit(build_features(data, columns), data.posted_rate/data.distance)


def predict_dollars(model, data, columns=None):
    values = model.predict(build_features(data, columns)) * data.distance.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite predictions')
    floor_count = int((values <= 0).sum())
    return np.maximum(values, 0.01), floor_count


def cross_validate(data, params=None, columns=None, baseline=None):
    rows = []
    for month in FOLD_MONTHS:
        start = pd.Period(month).start_time
        end = (pd.Period(month)+1).start_time
        train = data.loc[data.date < start]
        valid = data.loc[(data.date >= start) & (data.date < end)]
        if not len(train) or not len(valid):
            raise ValueError(f'Empty fold {month}')
        if baseline == 'quote':
            predictions = valid.quote_signal * valid.distance
        elif baseline == 'median':
            predictions = (train.posted_rate/train.distance).median() * valid.distance
        else:
            model = fit_model(train, params, columns)
            predictions, _ = predict_dollars(model, valid, columns)
        rows.append(dict(month=month, train_rows=len(train), validation_rows=len(valid),
                         **metrics(valid.posted_rate, predictions)))
    return pd.DataFrame(rows)
