from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CUTOFF = '2025-09-01'
FOLD_MONTHS = ['2025-04', '2025-05', '2025-06', '2025-07', '2025-08']
CORE_FEATURES = ['distance', 'weight', 'weight_missing', 'weight_was_negative',
                 'eq_DryVan', 'eq_Flatbed', 'eq_Reefer', 'pickup_lat', 'pickup_lon',
                 'delivery_lat', 'delivery_lon']
FEATURES = CORE_FEATURES + ['market_index', 'market_index_missing']
QUOTE_FEATURES = ['quote_signal', 'day_qs_median', 'day_low_qs_share']
PARAMS = dict(objective='huber', alpha=1.0, n_estimators=600, learning_rate=0.03,
              num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=SEED, verbosity=-1, n_jobs=2)
DISTANCE_BINS = [0, 200, 300, 450, 600, 1000, float('inf')]
DISTANCE_LABELS = ['<=200', '200-300', '300-450', '450-600', '600-1000', '>1000']
