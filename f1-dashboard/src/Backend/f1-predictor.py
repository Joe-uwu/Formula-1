from flask import Flask, jsonify, request
from f1predictor_functions import F1Predictor
import pandas as pd

app = Flask(__name__)
predictor = F1Predictor()

@app.route('/api/current_season', methods=['GET'])
def get_current_season():
    data = predictor.get_current_season()
    return jsonify(data)

@app.route('/api/driver_standings', methods=['GET'])
def get_driver_standings():
    data = predictor.get_driver_standings()
    return jsonify(data)

@app.route('/api/constructor_standings', methods=['GET'])
def get_constructor_standings():
    data = predictor.get_constructor_standings()
    return jsonify(data)

@app.route('/api/upcoming_races', methods=['GET'])
def get_upcoming_races():
    data = predictor.get_upcoming_races()
    return jsonify(data)

@app.route('/api/predict_race', methods=['POST'])
def predict_race():
    data = request.json

    required_features = ['circuit', 'driver', 'constructor', 'grid_position', 'qualifying_time',
                         'year', 'month', 'dayofyear', 'avg_finish_driver', 'avg_finish_constructor',
                         'win_ratio_driver', 'win_ratio_constructor']
    
    missing_features = [feat for feat in required_features if feat not in data]
    if missing_features:
        return jsonify({'error': f'Missing required features: {", ".join(missing_features)}'}), 400
    
    try:
        prediction = predictor.predict_race(data)
        return jsonify({'predicted_position': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    if predictor.prediction_model is None:
        return jsonify({'error': 'Model has not been trained yet'}), 400
    
    feature_importance = pd.DataFrame({
        'feature': predictor.prediction_model.feature_names_in_,
        'importance': predictor.prediction_model.feature_importances_
    }).sort_values('importance', ascending=False).to_dict('records')
    
    return jsonify({
        'model_type': type(predictor.prediction_model).__name__,
        'feature_importance': feature_importance,
        'model_params': predictor.prediction_model.get_params()
    })

@app.route('/api/train_model', methods=['POST'])
def train_model():
    try:
        predictor.train_prediction_model()
        return jsonify({'message': 'Model trained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)