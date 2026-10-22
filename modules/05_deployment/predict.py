import logging
import pickle

from flask import Flask
from flask import request
from flask import jsonify


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask('churn_service')

C = 1.0
MODEL_FILE = f'model_C={C}.bin'


def load_model(path):
    logger.info('Loading model from %s', path)
    with open(path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    
    logger.info('Model loaded')
    return dv, model


# load once at startup (module level) so the route can use them
dv, model = load_model(MODEL_FILE)

# POST means client sends data
@app.route('/predict', methods=['POST'])
def predict():
    # Read the customer data (JSON) sent by the client
    customer = request.get_json()
    logger.info('Received prediction request')

    try:
        X = dv.transform([customer])
        y_pred = model.predict_proba(X)[0, 1]
    except Exception:
        logger.exception('Prediction failed')   # logs the full traceback
        return jsonify({'error': 'invalid input'}), 400

    churn = y_pred >= 0.5
    logger.info('Churn probability: %.3f (churn=%s)', y_pred, churn)

    response = {
        'churn': bool(churn),
        'churn_probability': float(y_pred)
    }

    # Send the dict back to the client as JSON
    return jsonify(response)


if __name__ == '__main__':  # only when run directly, not when imported
    app.run(
        debug=True,      # auto-reload and show detailed errors
        host='0.0.0.0',  # accept connections from any machine
        port=9696        # port to listen on
    )
