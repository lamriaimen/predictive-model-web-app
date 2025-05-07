from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Mock prediction function (replace with actual model predictions later)
def mock_predict(values, model_type):
    # This is a simple mock that returns the average of input values plus some random variation
    avg = np.mean(values)
    return float(avg + np.random.normal(0, 0.1))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        values = [float(x) for x in data['values']]
        model_type = data['model_type']
        
        # Get prediction using mock function
        next_hour = mock_predict(values, model_type)
        
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
