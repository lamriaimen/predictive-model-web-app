from flask import Flask, render_template, request, jsonify
import numpy as np
import numpy._core  # ensure joblib can unpickle scalers
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

# Load your models
models = {
    'PM2.5': tf.keras.models.load_model('models/PM25_Forecaster.keras'),
    'C6H6':  tf.keras.models.load_model('models/C6H6_Forecaster.keras'),
    'CO':    tf.keras.models.load_model('models/CO_Forecaster.keras'),
    'NO2':   tf.keras.models.load_model('models/NO2_Forecaster.keras'),
}

scalers = {
    'PM2.5': joblib.load('scalers/PM25_scaler.save'),
    'C6H6':  joblib.load('scalers/C6H6_scaler.save'),
    'CO':    joblib.load('scalers/CO_scaler.save'),
    'NO2':   joblib.load('scalers/NO2_scaler.save'),
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data       = request.json
        values     = np.array([float(x) for x in data['values']], dtype=np.float32)
        model_type = data['model_type']
        model      = models.get(model_type)
        scaler     = scalers.get(model_type)

        if model is None:
            return jsonify(success=False, error='Invalid model type')

        # 1) scale the batch to [0,1]
        col        = values.reshape(-1,1)
        scaled_col = scaler.fit_transform(col)
        X          = scaled_col.reshape(1, 10, 1)

        # 2) predict in normalized space
        pred_norm_np = model.predict(X).flatten()[0]

        # 3) inverse‐scale back to original units
        pred_orig_np = scaler.inverse_transform([[pred_norm_np]])[0,0]

        # cast to native Python
        pred = float(pred_orig_np)

        response = {
            'success': True,
            'prediction': pred  # this is your “real-world” prediction
        }

        # 4) optional RMSE if real_value provided
        if data.get('real_value') is not None:
            real_val = float(data['real_value'])
            # RMSE for single sample = abs error
            rmse = float(np.sqrt((pred - real_val)**2))
            response['rmse'] = rmse

        return jsonify(response)

    except Exception as e:
        return jsonify(success=False, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
