from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load all models
models = {
    'PM2.5': tf.keras.models.load_model('models/PM25_Forecaster.keras'),
    'C6H6': tf.keras.models.load_model('models/C6H6_Forecaster.keras'),
    'CO': tf.keras.models.load_model('models/CO_Forecaster.keras'),
    'NO2': tf.keras.models.load_model('models/NO2_Forecaster.keras')
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        values = np.array([float(x) for x in data['values']], dtype=np.float32)
        model_type = data['model_type']
        
        # Reshape input to (1, 10, 1)
        X = values.reshape(1, 10, 1)
        
        # Get prediction from selected model
        model = models.get(model_type)
        if model is None:
            return jsonify({'success': False, 'error': 'Invalid model type'})
        
        pred = model.predict(X)
        next_hour = float(pred.flatten()[0])
        
        return jsonify({
            'success': True,
            'prediction': next_hour
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
