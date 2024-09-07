import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class F1Predictor:
    def __init__(self):
        self.ERGAST_API_BASE = "http://ergast.com/api/f1"
        self.prediction_model = None
        self.label_encoder = None
        self.scaler = StandardScaler()

    def prepare_data_for_model(self):
        seasons = range(2019, 2024)
        data = []
        for season in seasons:
            season_data = self.fetch_ergast_data(f"{season}/results")
            data.extend(season_data['MRData']['RaceTable']['Races'])
        
        df = pd.DataFrame(data)
        
        # Feature engineering
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['dayofyear'] = pd.to_datetime(df['date']).dt.dayofyear
        
        # Calculate average finish position for drivers and constructors
        df['avg_finish_driver'] = df.groupby('driver')['position'].transform('mean')
        df['avg_finish_constructor'] = df.groupby('constructor')['position'].transform('mean')
        
        # Calculate win ratio for drivers and constructors
        df['win_ratio_driver'] = df.groupby('driver')['position'].transform(lambda x: (x == 1).mean())
        df['win_ratio_constructor'] = df.groupby('constructor')['position'].transform(lambda x: (x == 1).mean())
        
        return df

    def train_prediction_model(self):
        df = self.prepare_data_for_model()
        
        features = ['circuit', 'driver', 'constructor', 'grid_position', 'qualifying_time',
                    'year', 'month', 'dayofyear', 'avg_finish_driver', 'avg_finish_constructor',
                    'win_ratio_driver', 'win_ratio_constructor']
        target = 'finish_position'
        
        X = df[features]
        y = df[target]
        
        le = LabelEncoder()
        for feature in ['circuit', 'driver', 'constructor']:
            X[feature] = le.fit_transform(X[feature])
        
        X = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        self.prediction_model = grid_search.best_estimator_
        self.label_encoder = le
        
        # Model evaluation
        y_pred = self.prediction_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R-squared Score: {r2}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.prediction_model, X, y, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {np.mean(cv_scores)}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.prediction_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Feature Importance:")
        print(feature_importance)

    def predict_race(self, data):
        if self.prediction_model is None or self.label_encoder is None:
            self.train_prediction_model()
        
        input_data = pd.DataFrame([data])
        for feature in ['circuit', 'driver', 'constructor']:
            input_data[feature] = self.label_encoder.transform(input_data[feature])
        
        input_data_scaled = self.scaler.transform(input_data)
        prediction = self.prediction_model.predict(input_data_scaled)
        return prediction[0]

    def fetch_ergast_data(self, endpoint):
        response = requests.get(f"{self.ERGAST_API_BASE}/{endpoint}.json")
        return response.json()

    def get_current_season(self):
        return self.fetch_ergast_data("current")

    def get_driver_standings(self):
        return self.fetch_ergast_data("current/driverStandings")

    def get_constructor_standings(self):
        return self.fetch_ergast_data("current/constructorStandings")

    def get_upcoming_races(self):
        return self.fetch_ergast_data("current")

    def prepare_data_for_model(self):
        seasons = range(2019, 2024)
        data = []
        for season in seasons:
            season_data = self.fetch_ergast_data(f"{season}/results")
            data.extend(season_data['MRData']['RaceTable']['Races'])
        
        df = pd.DataFrame(data)
        return df

    def train_prediction_model(self):
        df = self.prepare_data_for_model()
        
        features = ['circuit', 'driver', 'constructor', 'grid_position', 'qualifying_time']
        target = 'finish_position'
        
        X = df[features]
        y = df[target]
        
        le = LabelEncoder()
        for feature in ['circuit', 'driver', 'constructor']:
            X[feature] = le.fit_transform(X[feature])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        self.prediction_model = model
        self.label_encoder = le

    def predict_race(self, data):
        if self.prediction_model is None or self.label_encoder is None:
            self.train_prediction_model()
        
        input_data = pd.DataFrame([data])
        for feature in ['circuit', 'driver', 'constructor']:
            input_data[feature] = self.label_encoder.transform(input_data[feature])
        
        prediction = self.prediction_model.predict(input_data)
        return prediction[0]