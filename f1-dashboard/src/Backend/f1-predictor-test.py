import requests
import pandas as pd
import numpy as np
import fastf1
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

class F1Predictor:
    def __init__(self):
        """
        Initializes the F1Predictor class.
        """
        fastf1.Cache.enable_cache('cache')  # Use 'cache' directory for caching
        self.base_url = "http://ergast.com/api/f1"
        self.current_year = datetime.now().year
        self.api_key = 'c5d27272ec454d7b9f6234316241009'  # Set your API key as an environment variable
        if not self.api_key:
            print("Warning: WEATHER_API_KEY environment variable not set. Please set it to use weather features.")

        self.grand_prix_data = {
            # Grand Prix data with actual dates
            "Bahrain": {"city": "Sakhir", "country": "BH", "date": "2023-03-05"},
            "Saudi Arabia": {"city": "Jeddah", "country": "SA", "date": "2023-03-19"},
            "Australia": {"city": "Melbourne", "country": "AU", "date": "2023-04-02"},
            "Azerbaijan": {"city": "Baku", "country": "AZ", "date": "2023-04-30"},
            "Miami": {"city": "Miami", "country": "US", "date": "2023-05-07"},
            "Monaco": {"city": "Monaco", "country": "MC", "date": "2023-05-28"},
            "Spain": {"city": "Barcelona", "country": "ES", "date": "2023-06-04"},
            "Canada": {"city": "Montreal", "country": "CA", "date": "2023-06-18"},
            "Austria": {"city": "Spielberg", "country": "AT", "date": "2023-07-02"},
            "Great Britain": {"city": "Silverstone", "country": "GB", "date": "2023-07-09"},
            "Hungary": {"city": "Budapest", "country": "HU", "date": "2023-07-23"},
            "Belgium": {"city": "Spa-Francorchamps", "country": "BE", "date": "2023-07-30"},
            "Netherlands": {"city": "Zandvoort", "country": "NL", "date": "2023-08-27"},
            "Italy": {"city": "Monza", "country": "IT", "date": "2023-09-03"},
            "Singapore": {"city": "Singapore", "country": "SG", "date": "2023-09-17"},
            "Japan": {"city": "Suzuka", "country": "JP", "date": "2023-09-24"},
            "Qatar": {"city": "Lusail", "country": "QA", "date": "2023-10-08"},
            "United States": {"city": "Austin", "country": "US", "date": "2023-10-22"},
            "Mexico": {"city": "Mexico City", "country": "MX", "date": "2023-10-29"},
            "Brazil": {"city": "Sao Paulo", "country": "BR", "date": "2023-11-05"},
            "Las Vegas": {"city": "Las Vegas", "country": "US", "date": "2023-11-18"},
            "Abu Dhabi": {"city": "Abu Dhabi", "country": "AE", "date": "2023-11-26"},
        }
        self.active_drivers = self.get_active_drivers()
        self.driver_standings = self.get_driver_standings()
        self.constructor_standings = self.get_constructor_standings()
        self.recent_race_results = self.get_recent_race_results(num_races=5)

    def get_active_drivers(self):
        """
        Retrieves the list of active drivers for the current season.
        """
        url = f"{self.base_url}/{self.current_year}/drivers.json?limit=100"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            drivers = data['MRData']['DriverTable']['Drivers']
            driver_list = []
            for driver in drivers:
                driver_list.append({
                    'DriverId': driver['driverId'],
                    'Driver': f"{driver['givenName']} {driver['familyName']}"
                })
            return pd.DataFrame(driver_list)
        else:
            print(f"Error fetching active drivers: {response.status_code}")
            return pd.DataFrame()

    def get_driver_standings(self):
        """
        Retrieves current driver standings.
        """
        url = f"{self.base_url}/{self.current_year}/driverStandings.json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            standings = data['MRData']['StandingsTable']['StandingsLists']
            standings_list = []
            if standings:
                standings = standings[0]['DriverStandings']
                for entry in standings:
                    driver = entry['Driver']
                    constructor = entry['Constructors'][0]
                    standings_list.append({
                        'DriverId': driver['driverId'],
                        'Driver': f"{driver['givenName']} {driver['familyName']}",
                        'ConstructorId': constructor['constructorId'],
                        'Points': float(entry['points']),
                        'Wins': int(entry['wins']),
                    })
            return pd.DataFrame(standings_list)
        else:
            print(f"Error fetching driver standings: {response.status_code}")
            return pd.DataFrame()

    def get_constructor_standings(self):
        """
        Retrieves current constructor standings.
        """
        url = f"{self.base_url}/{self.current_year}/constructorStandings.json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            standings = data['MRData']['StandingsTable']['StandingsLists']
            standings_list = []
            if standings:
                standings = standings[0]['ConstructorStandings']
                for entry in standings:
                    constructor = entry['Constructor']
                    standings_list.append({
                        'ConstructorId': constructor['constructorId'],
                        'ConstructorName': constructor['name'],
                        'ConstructorPoints': float(entry['points']),
                        'ConstructorWins': int(entry['wins']),
                    })
            return pd.DataFrame(standings_list)
        else:
            print(f"Error fetching constructor standings: {response.status_code}")
            return pd.DataFrame()

    def get_historical_race_results(self, circuit_id=None):
        """
        Retrieves historical race results for the past 3 years.
        """
        results = []
        # Collect data from the last 3 seasons
        for year in range(self.current_year - 3, self.current_year):
            # Fetch all races in the season
            season_url = f"{self.base_url}/{year}.json"
            response = requests.get(season_url)
            if response.status_code == 200:
                data = response.json()
                races = data['MRData']['RaceTable']['Races']
                for race in races:
                    round_number = race['round']
                    # Skip if circuit_id is specified and doesn't match
                    if circuit_id and race['Circuit']['circuitId'] != circuit_id:
                        continue
                    # Fetch results for each race
                    url = f"{self.base_url}/{year}/{round_number}/results.json?limit=1000"
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        race_results = data['MRData']['RaceTable']['Races'][0]
                        for result in race_results['Results']:
                            driver = result['Driver']
                            constructor = result['Constructor']
                            results.append({
                                'Year': int(year),
                                'RaceName': race_results['raceName'],
                                'CircuitId': race_results['Circuit']['circuitId'],
                                'DriverId': driver['driverId'],
                                'Driver': f"{driver['givenName']} {driver['familyName']}",
                                'ConstructorId': constructor['constructorId'],
                                'GridPosition': int(result['grid']) if result['grid'].isdigit() else None,
                                'FinishPosition': int(result['position']) if result['position'].isdigit() else None,
                                'Status': result['status'],
                                'Points': float(result['points']),
                            })
                    else:
                        print(f"Error fetching results for round {round_number} in {year}: {response.status_code}")
            else:
                print(f"Error fetching race schedule for {year}: {response.status_code}")
        return pd.DataFrame(results)

    def get_circuit_id(self, track_name):
        circuit_mapping = {
            "bahrain": "bahrain",
            "saudi arabia": "jeddah",
            "australia": "albert_park",
            "azerbaijan": "baku",
            "miami": "miami",
            "monaco": "monaco",
            "spain": "catalunya",
            "canada": "villeneuve",
            "austria": "red_bull_ring",
            "great britain": "silverstone",
            "britain": "silverstone",
            "hungary": "hungaroring",
            "belgium": "spa",
            "netherlands": "zandvoort",
            "italy": "monza",
            "singapore": "marina_bay",
            "japan": "suzuka",
            "qatar": "losail",  # Note: circuitId is 'losail' for Qatar
            "united states": "cota",
            "mexico": "rodriguez",
            "brazil": "interlagos",
            "las vegas": "las_vegas",
            "abu dhabi": "yas_marina",
        }
        return circuit_mapping.get(track_name.lower(), track_name.lower())

    def get_recent_performance(self):
        """
        Retrieves recent performance of drivers for the current season.
        """
        standings = self.driver_standings
        if not standings.empty:
            return standings.rename(columns={'Points': 'Points', 'Wins': 'Wins'})
        else:
            return pd.DataFrame()

    def get_weather_forecast(self, city, country, date):
        """
        Fetches weather forecast for a specified location and date.
        """
        base_url = "http://api.weatherapi.com/v1/forecast.json"
        days_from_now = (date - datetime.now().date()).days
        if days_from_now < 0 or days_from_now > 14 or not self.api_key:
            # Return default values
            return {
                "temperature": 25,
                "description": "Unknown",
                "is_wet": False,
                "rain_chance": 0
            }
        params = {
            "key": self.api_key,
            "q": f"{city},{country}",
            "dt": date.strftime("%Y-%m-%d"),
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            forecast_day = data['forecast']['forecastday'][0]
            return {
                "temperature": forecast_day['day']['avgtemp_c'],
                "description": forecast_day['day']['condition']['text'],
                "is_wet": forecast_day['day']['daily_will_it_rain'] == 1,
                "rain_chance": float(forecast_day['day']['daily_chance_of_rain']) / 100.0
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return {
                "temperature": 25,
                "description": "Unknown",
                "is_wet": False,
                "rain_chance": 0
            }

    def get_recent_race_results(self, num_races=5):
        """
        Retrieves results from the last few completed races.
        """
        results = []
        # Fetch the race schedule for the current season
        schedule_url = f"{self.base_url}/{self.current_year}.json"
        response = requests.get(schedule_url)
        if response.status_code == 200:
            data = response.json()
            races = data['MRData']['RaceTable']['Races']
            # Filter races that have already occurred
            today = datetime.now()
            completed_races = []
            for race in races:
                race_date = datetime.strptime(race['date'], '%Y-%m-%d')
                if race_date < today:
                    completed_races.append(race)
            # Get the last num_races races
            recent_races = completed_races[-num_races:]
            for race in recent_races:
                round_number = race['round']
                url = f"{self.base_url}/{self.current_year}/{round_number}/results.json?limit=1000"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    race_results = data['MRData']['RaceTable']['Races'][0]
                    for result in race_results['Results']:
                        driver = result['Driver']
                        constructor = result['Constructor']
                        results.append({
                            'RaceName': race_results['raceName'],
                            'DriverId': driver['driverId'],
                            'ConstructorId': constructor['constructorId'],
                            'Points': float(result['points']),
                            'FinishPosition': int(result['position']) if result['position'].isdigit() else None,
                        })
                else:
                    print(f"Error fetching results for round {round_number}: {response.status_code}")
        else:
            print(f"Error fetching race schedule: {response.status_code}")
        return pd.DataFrame(results)

    def get_qualifying_results(self, grand_prix_name, year):
        """
        Retrieves qualifying results for a specific Grand Prix.
        """
        circuit_id = self.get_circuit_id(grand_prix_name)
        url = f"{self.base_url}/{year}/circuits/{circuit_id}/qualifying.json?limit=1000"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            races = data['MRData']['RaceTable']['Races']
            if races:
                qualifying_results = races[0]['QualifyingResults']
                results = []
                for result in qualifying_results:
                    driver = result['Driver']
                    results.append({
                        'DriverId': driver['driverId'],
                        'QualifyingPosition': int(result['position'])
                    })
                return pd.DataFrame(results)
            else:
                # No qualifying data available
                return pd.DataFrame()
        else:
            print(f"Error fetching qualifying results: {response.status_code}")
            return pd.DataFrame()

    def get_circuit_features(self, circuit_id):
        """
        Returns circuit-specific features.
        """
        # For this example, we'll use a simple mapping. In a real scenario, you'd have more detailed data.
        circuit_features = {
            'bahrain': {'CircuitType': 'Permanent', 'CircuitLength': 5.412},
            'jeddah': {'CircuitType': 'Street', 'CircuitLength': 6.174},
            'albert_park': {'CircuitType': 'Street', 'CircuitLength': 5.278},
            'baku': {'CircuitType': 'Street', 'CircuitLength': 6.003},
            'miami': {'CircuitType': 'Street', 'CircuitLength': 5.412},
            'monaco': {'CircuitType': 'Street', 'CircuitLength': 3.337},
            'catalunya': {'CircuitType': 'Permanent', 'CircuitLength': 4.655},
            'villeneuve': {'CircuitType': 'Street', 'CircuitLength': 4.361},
            'red_bull_ring': {'CircuitType': 'Permanent', 'CircuitLength': 4.318},
            'silverstone': {'CircuitType': 'Permanent', 'CircuitLength': 5.891},
            'hungaroring': {'CircuitType': 'Permanent', 'CircuitLength': 4.381},
            'spa': {'CircuitType': 'Permanent', 'CircuitLength': 7.004},
            'zandvoort': {'CircuitType': 'Permanent', 'CircuitLength': 4.259},
            'monza': {'CircuitType': 'Permanent', 'CircuitLength': 5.793},
            'marina_bay': {'CircuitType': 'Street', 'CircuitLength': 5.063},
            'suzuka': {'CircuitType': 'Permanent', 'CircuitLength': 5.807},
            'losail': {'CircuitType': 'Permanent', 'CircuitLength': 5.380},
            'cota': {'CircuitType': 'Permanent', 'CircuitLength': 5.513},
            'rodriguez': {'CircuitType': 'Permanent', 'CircuitLength': 4.304},
            'interlagos': {'CircuitType': 'Permanent', 'CircuitLength': 4.309},
            'las_vegas': {'CircuitType': 'Street', 'CircuitLength': 6.120},
            'yas_marina': {'CircuitType': 'Permanent', 'CircuitLength': 5.554},
        }
        return circuit_features.get(circuit_id, {'CircuitType': 'Unknown', 'CircuitLength': 5.0})

    def predict_race_winner(self, grand_prix_name, race_year):
        """
        Predicts the winner of a specified Grand Prix.
        """
        circuit_id = self.get_circuit_id(grand_prix_name)

        # Get race details
        race_info = self.grand_prix_data.get(grand_prix_name)
        if not race_info:
            print(f"No data available for {grand_prix_name}")
            return None

        # Get weather forecast
        city = race_info['city']
        country = race_info['country']
        race_date_str = race_info['date']
        race_date = datetime.strptime(race_date_str, "%Y-%m-%d").date()
        weather = self.get_weather_forecast(city, country, race_date)

        # Get historical race results
        historical_results = self.get_historical_race_results()

        if historical_results.empty:
            print("No historical data available")
            return None

        # Filter active drivers
        active_driver_ids = self.active_drivers['DriverId'].unique()
        historical_results = historical_results[historical_results['DriverId'].isin(active_driver_ids)]

        # Driver's performance at the specific circuit
        circuit_results = self.get_historical_race_results(circuit_id=circuit_id)
        if circuit_results.empty:
            # If no data for circuit, use overall performance
            circuit_results = historical_results.copy()

        # Calculate circuit-specific stats
        circuit_stats = circuit_results.groupby('DriverId').agg(
            AvgFinishPositionCircuit=('FinishPosition', 'mean'),
            BestFinishPositionCircuit=('FinishPosition', 'min'),
            TotalPointsCircuit=('Points', 'sum'),
            WinsCircuit=('FinishPosition', lambda x: (x == 1).sum())
        ).reset_index()

        # Merge data
        data = self.driver_standings.merge(circuit_stats, on='DriverId', how='left')
        data = data.merge(self.constructor_standings[['ConstructorId', 'ConstructorName', 'ConstructorPoints', 'ConstructorWins']], on='ConstructorId', how='left')

        # Add qualifying data if available
        qualifying_results = self.get_qualifying_results(grand_prix_name, race_year)
        if not qualifying_results.empty:
            data = data.merge(qualifying_results[['DriverId', 'QualifyingPosition']], on='DriverId', how='left')
        else:
            # Use average grid positions if qualifying data not available
            avg_grid_positions = historical_results.groupby('DriverId')['GridPosition'].mean().reset_index()
            avg_grid_positions.rename(columns={'GridPosition': 'QualifyingPosition'}, inplace=True)
            data = data.merge(avg_grid_positions, on='DriverId', how='left')

        # Add weather data
        data['Temperature'] = weather['temperature']
        data['IsWet'] = int(weather['is_wet'])
        data['RainChance'] = weather['rain_chance']

        # Add circuit features
        circuit_features = self.get_circuit_features(circuit_id)
        data['CircuitType'] = circuit_features['CircuitType']
        data['CircuitLength'] = circuit_features['CircuitLength']

        # Convert categorical features to numerical
        data['CircuitType'] = data['CircuitType'].map({'Permanent': 0, 'Street': 1, 'Unknown': 2})

        # Fill missing values
        data.fillna(0, inplace=True)

        # Feature Engineering
        num_years_circuit = circuit_results['Year'].nunique() if circuit_results['Year'].nunique() > 0 else 1
        data['WinRateCircuit'] = data['WinsCircuit'] / num_years_circuit

        # **Prepare features (ensure they match training features)**
        feature_columns = [
            'QualifyingPosition',
            'Wins',
            'Points',
            'ConstructorPoints',
            'ConstructorWins',
            'AvgFinishPositionCircuit',
            'BestFinishPositionCircuit',
            'TotalPointsCircuit',
            'WinsCircuit',
            'WinRateCircuit',
            'CircuitType',
            'CircuitLength',
            'Temperature',
            'IsWet',
            'RainChance',
        ]
        features = data[feature_columns]

        # **Prepare historical features (include all features)**
        # For features not available in historical data, fill with default values
        historical_data = historical_results.merge(
            self.driver_standings[['DriverId', 'ConstructorId', 'Wins', 'Points']],
            on='DriverId',
            how='left',
            suffixes=('', '_standings')
        )

        # Use 'ConstructorId' from 'historical_results'
        # For consistency, we can proceed with existing 'ConstructorId'

        historical_data = historical_data.merge(
            self.constructor_standings[['ConstructorId', 'ConstructorPoints', 'ConstructorWins']],
            on='ConstructorId',
            how='left'
        )

        # Add circuit stats to historical data
        historical_data = historical_data.merge(circuit_stats, on='DriverId', how='left')

        # Add qualifying data (use 'GridPosition' as 'QualifyingPosition')
        historical_data['QualifyingPosition'] = historical_data['GridPosition']

        # Add weather data (use default values)
        historical_data['Temperature'] = 25  # Default temperature
        historical_data['IsWet'] = 0         # Assume dry conditions
        historical_data['RainChance'] = 0.0  # No rain

        # Add circuit features
        historical_data['CircuitType'] = historical_data['CircuitId'].map(lambda x: self.get_circuit_features(x)['CircuitType']).map({'Permanent': 0, 'Street': 1, 'Unknown': 2})
        historical_data['CircuitLength'] = historical_data['CircuitId'].map(lambda x: self.get_circuit_features(x)['CircuitLength'])

        # Calculate 'WinRateCircuit' (use 1 as default for 'num_years_circuit')
        historical_data['WinRateCircuit'] = historical_data['WinsCircuit'] / num_years_circuit

        # Prepare features for historical data (ensure feature columns match)
        historical_features = historical_data[feature_columns].fillna(0)

        # Target variable
        historical_data['Win'] = historical_data['FinishPosition'] == 1
        target = historical_data['Win']

        # Handle missing values and infinite values
        historical_features = historical_features.replace([np.inf, -np.inf], np.nan).fillna(0)

        # **Scale features**
        scaler = StandardScaler()
        scaler.fit(historical_features)

        # Scale training features
        historical_features_scaled = scaler.transform(historical_features)

        print("Sample historical features:")
        print(historical_features.head())
        print("Sample prediction features:")
        print(features.head())

        # Handle class imbalance
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(historical_features_scaled, target)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        # Model training using XGBoost
        model = XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Prepare data for prediction
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        features_scaled = scaler.transform(features)

        probabilities = model.predict_proba(features_scaled)[:, 1]

        # Assign probabilities to drivers
        data['WinProbability'] = probabilities * 100  # Convert to percentage

        # Sort drivers by descending probability
        data.sort_values('WinProbability', ascending=False, inplace=True)

        # Feature Importances
        importances = model.feature_importances_
        feature_importances = pd.Series(importances, index=feature_columns)
        print("Feature Importances:")
        print(feature_importances.sort_values(ascending=False))

        # Return the DataFrame containing drivers and their win probabilities
        return data[['Driver', 'ConstructorName', 'WinProbability']]

# Example usage
if __name__ == "__main__":
    predictor = F1Predictor()
    grand_prix = 'Singapore'  # Grand Prix name (e.g., 'Abu Dhabi', 'Monaco')
    race_year = 2024         # Year of the race

    driver_probabilities = predictor.predict_race_winner(grand_prix, race_year)
    if driver_probabilities is not None:
        print(f"Predicted probabilities for the {grand_prix} Grand Prix {race_year}:")
        print(driver_probabilities.to_string(index=False))
    else:
        print("Prediction could not be made due to insufficient data.")