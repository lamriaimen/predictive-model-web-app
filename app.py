import os
import logging
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
import joblib

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable GPU usage (Render doesn’t support it)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

app = Flask(__name__)

# Load TFLite interpreters and scalers
interpreters = {
    'PM2.5': tf.lite.Interpreter(model_path='models/PM25_Forecaster.tflite'),
    'C6H6':  tf.lite.Interpreter(model_path='models/C6H6_Forecaster.tflite'),
    'CO':    tf.lite.Interpreter(model_path='models/CO_Forecaster.tflite'),
    'NO2':   tf.lite.Interpreter(model_path='models/NO2_Forecaster.tflite'),
}

scalers = {
    'PM2.5': joblib.load('scalers/PM25_scaler.save'),
    'C6H6':  joblib.load('scalers/C6H6_scaler.save'),
    'CO':    joblib.load('scalers/CO_scaler.save'),
    'NO2':   joblib.load('scalers/NO2_scaler.save'),
}

# Allocate tensors for interpreters
for interpreter in interpreters.values():
    interpreter.allocate_tensors()

# ATMO thresholds (unchanged)
ATMO_THRESHOLDS = {
    'PM2.5': [(0, 10, "Bon"), (10, 20, "Moyen"), (20, 25, "Dégradé"), (25, 50, "Mauvais"), (50, 75, "Très mauvais"), (75, np.inf, "Extrêmement mauvais")],
    'PM10':  [(0, 20, "Bon"), (20, 40, "Moyen"), (40, 50, "Dégradé"), (50,100, "Mauvais"), (100,150,"Très mauvais"), (150,np.inf,"Extrêmement mauvais")],
    'NO2':   [(0, 40, "Bon"), (40, 90, "Moyen"), (90,120, "Dégradé"), (120,230,"Mauvais"), (230,340,"Très mauvais"), (340,np.inf,"Extrêmement mauvais")],
}

def atmo_status(pollutant: str, value: float) -> str:
    """Retourne l’étiquette ATMO en français pour le polluant donné."""
    if pollutant not in ATMO_THRESHOLDS:
        return "Indisponible"
    for low, high, label in ATMO_THRESHOLDS[pollutant]:
        if low <= value < high:
            return label
    return "Indisponible"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        values = np.array([float(x) for x in data['values']], dtype=np.float32)
        model_type = data['model_type']

        interpreter = interpreters.get(model_type)
        scaler = scalers.get(model_type)
        if interpreter is None or scaler is None:
            return jsonify(success=False, error='Invalid model type')

        # 1) Mise à l’échelle
        col = values.reshape(-1, 1)
        scaled_col = scaler.transform(col)
        X = scaled_col.reshape(1, 10, 1).astype(np.float32)  # TFLite requires float32

        # 2) Prédiction avec l’interpréteur TFLite
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], X)
        interpreter.invoke()
        pred_norm_np = interpreter.get_tensor(output_details[0]['index']).flatten()[0]

        # 3) Remise à l’échelle d’origine
        pred_orig_np = scaler.inverse_transform([[pred_norm_np]])[0, 0]
        pred = float(pred_orig_np)

        # 4) Calcul du statut ATMO
        status = atmo_status(model_type if model_type != "PM2.5" else "PM2.5", pred)

        response = {
            'success': True,
            'prediction': pred,
            'status_fr': status
        }

        # 5) Optionnel : RMSE si real_value fourni
        if data.get('real_value') is not None:
            real_val = float(data['real_value'])
            response['rmse'] = abs(pred - real_val)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify(success=False, error=str(e))

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
