from flask import Flask, render_template, request, jsonify
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

# Load your models
models = {
    'PM2.5': keras.saving.load_model('models/PM25_Forecaster.keras', compile=False),
    'C6H6':  keras.saving.load_model('models/C6H6_Forecaster.keras', compile=False),
    'CO':    keras.saving.load_model('models/CO_Forecaster.keras', compile=False),
    'NO2':   keras.saving.load_model('models/NO2_Forecaster.keras', compile=False),
}

scalers = {
    'PM2.5': joblib.load('scalers/PM25_scaler.save'),
    'C6H6':  joblib.load('scalers/C6H6_scaler.save'),
    'CO':    joblib.load('scalers/CO_scaler.save'),
    'NO2':   joblib.load('scalers/NO2_scaler.save'),
}
ATMO_THRESHOLDS = {
    'PM2.5':  [(0, 10,  "Bon"),
               (10, 20, "Moyen"),
               (20, 25, "Dégradé"),
               (25, 50, "Mauvais"),
               (50, 75, "Très mauvais"),
               (75, np.inf, "Extrêmement mauvais")],

    'PM10':   [(0, 20,  "Bon"),
               (20, 40, "Moyen"),
               (40, 50, "Dégradé"),
               (50,100, "Mauvais"),
               (100,150,"Très mauvais"),
               (150,np.inf,"Extrêmement mauvais")],

    'NO2':    [(0, 40,  "Bon"),
               (40, 90, "Moyen"),
               (90,120, "Dégradé"),
               (120,230,"Mauvais"),
               (230,340,"Très mauvais"),
               (340,np.inf,"Extrêmement mauvais")],

    # Vous n’avez pas de modèle O3 ou SO2 ici,
    # mais on pourrait les ajouter de la même façon
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
        data       = request.json
        values     = np.array([float(x) for x in data['values']], dtype=np.float32)
        model_type = data['model_type']
        model      = models.get(model_type)
        scaler     = scalers.get(model_type)
        if model is None:
            return jsonify(success=False, error='Invalid model type')

        # 1) mise à l’échelle
        col        = values.reshape(-1,1)
        scaled_col = scaler.fit_transform(col)
        X          = scaled_col.reshape(1, 10, 1)

        # 2) prédiction normalisée
        pred_norm_np = model.predict(X, verbose=0).flatten()[0]

        # 3) remise à l’échelle d’origine
        pred_orig_np = scaler.inverse_transform([[pred_norm_np]])[0,0]
        pred         = float(pred_orig_np)

        # 4) calcul du statut ATMO
        status = atmo_status(model_type if model_type!="PM2.5" else "PM2.5",
                             pred)

        response = {
            'success'    : True,
            'prediction' : pred,      # valeur brute (µg/m³)
            'status_fr'  : status     # étiquette qualitative
        }

        # 5) optionnel : RMSE si real_value fourni
        if data.get('real_value') is not None:
            real_val      = float(data['real_value'])
            response['rmse'] = abs(pred - real_val)  # identique à RMSE 1 échantillon

        return jsonify(response)

    except Exception as e:
        return jsonify(success=False, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
