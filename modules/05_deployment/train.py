import pickle
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

# Force numbers to display as decimals, not scientific notation
np.set_printoptions(suppress=True)

RANDOM_SEED = 1
C = 1.0
N_SPLITS = 5
OUTPUT_FILE = f'model_C={C}.bin'

# Load the dataset
df = pd.read_csv('../03_classification/churn_data.csv')
logger.info('Loaded %d rows from churn_data.csv', len(df))

# Clean the data
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'str'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

# Split the data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
logger.info('Split data: %d train, %d test', len(df_full_train), len(df_test))

# Define train and predict functions
numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# Cross-validation
logger.info('Starting %d-fold cross-validation with C=%s', N_SPLITS, C)

kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

scores = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(df_full_train), start=1):

    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)

    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    logger.info('Fold %d/%d: AUC=%.3f', fold, N_SPLITS, auc)

logger.info(
    'Cross-validation done: C=%s AUC=%.3f +- %.3f', C, 
    np.mean(scores), np.std(scores)
)

# Compute AUC
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
logger.info('Training final model on the full training set')

y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values

auc = roc_auc_score(y_test, y_pred)
logger.info('Test AUC: %.3f', auc)

# Save the model
with open(OUTPUT_FILE, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

logger.info('Model saved to %s', OUTPUT_FILE)
