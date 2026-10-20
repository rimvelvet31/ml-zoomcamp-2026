import logging
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

C = 1.0
MODEL_FILE = f'model_C={C}.bin'


def load_model(path):
    with open(path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def predict_single(customer, dv, model):
    X = dv.transform([customer])
    return model.predict_proba(X)[0, 1]


customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


if __name__ == '__main__':
    logger.info('Loading model from %s', MODEL_FILE)
    dv, model = load_model(MODEL_FILE)
    logger.info('Model loaded')

    churn_probability = predict_single(customer, dv, model)
    logger.info('Churn probability: %.3f', churn_probability)
