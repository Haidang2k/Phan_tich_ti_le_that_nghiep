import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
import warnings
warnings.filterwarnings("ignore")

class LSTMPredictor:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, data_path="global_unemployment_data.csv"):
        if not self._initialized:
            print("Initializing LSTM Predictor...")
            self.data_path = data_path
            self.model = None
            self.scaler = MinMaxScaler()
            
            # Fixed values for categories
            self.sex_values = ['Total', 'Male', 'Female']
            self.age_group_values = ['Total', 'Under 15', '15-24', '25+']
            
            # Load data first
            self.df = self._load_data()
            if self.df is not None:
                # Initialize and fit encoders with actual data
                self._init_encoders()
                # Train model
                self._train_model()
            
            self._initialized = True
    
    def _load_data(self):
        """Load the dataset"""
        try:
            print("Loading data...")
            df = pd.read_csv(self.data_path)
            # Ensure all year columns are numeric
            year_columns = [str(year) for year in range(2014, 2025)]
            for col in year_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"Data loaded successfully with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def _init_encoders(self):
        """Initialize and fit label encoders with actual data"""
        try:
            print("Initializing and fitting encoders...")
            
            # Country encoder - fit with actual unique countries from data
            self.country_encoder = LabelEncoder()
            unique_countries = sorted(self.df['country_name'].unique())
            self.country_encoder.fit(unique_countries)
            print(f"Fitted country encoder with {len(unique_countries)} unique countries")
            
            # Sex encoder - fit with actual unique sex values from data
            self.sex_encoder = LabelEncoder()
            unique_sex = sorted(set(self.sex_values) & set(self.df['sex'].unique()))
            self.sex_encoder.fit(unique_sex)
            print(f"Fitted sex encoder with values: {unique_sex}")
            
            # Age group encoder - fit with actual unique age groups from data
            self.age_encoder = LabelEncoder()
            unique_age_groups = sorted(set(self.age_group_values) & set(self.df['age_group'].unique()))
            self.age_encoder.fit(unique_age_groups)
            print(f"Fitted age group encoder with values: {unique_age_groups}")
            
            print("All encoders initialized and fitted successfully")
        except Exception as e:
            print(f"Error initializing encoders: {str(e)}")
            raise
    
    def _prepare_sequences(self):
        """Prepare sequences for training"""
        sequences = []
        targets = []
        
        for country in self.df['country_name'].unique():
            for sex in self.sex_values:
                for age in self.age_group_values:
                    mask = (self.df['country_name'] == country) & \
                           (self.df['sex'] == sex) & \
                           (self.df['age_group'] == age)
                    
                    if not mask.any():
                        continue
                    
                    row = self.df[mask].iloc[0]
                    values = []
                    for year in range(2020, 2025):  # Use last 5 years
                        value = row[str(year)]
                        if pd.isna(value):
                            break
                        values.append(value)
                    
                    if len(values) == 5:  # Only use complete sequences
                        values = np.array(values)
                        scaled = self.scaler.fit_transform(values.reshape(-1, 1))
                        sequences.append(scaled)
                        # Use the last value as target for training
                        targets.append(scaled[-1])
        
        if not sequences:
            raise ValueError("No valid sequences found for training")
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        print(f"Prepared {len(sequences)} training sequences")
        return sequences, targets
    
    def _build_model(self):
        """Build LSTM model"""
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(5, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def _train_model(self):
        """Train the LSTM model"""
        try:
            print("Preparing training data...")
            X, y = self._prepare_sequences()
            
            print("Training model...")
            self.model = self._build_model()
            history = self.model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            print(f"Model trained successfully. Final loss: {history.history['loss'][-1]:.4f}")
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise
    
    def predict(self, country, sex, age_group, future_years):
        """Make predictions for given parameters"""
        try:
            # Input validation
            if country not in self.df['country_name'].unique():
                raise ValueError(f"Invalid country: {country}")
            if sex not in self.sex_values:
                raise ValueError(f"Invalid sex: {sex}")
            if age_group not in self.age_group_values:
                raise ValueError(f"Invalid age group: {age_group}")
            
            # Get historical data
            mask = (self.df['country_name'] == country) & \
                   (self.df['sex'] == sex) & \
                   (self.df['age_group'] == age_group)
            
            if not mask.any():
                raise ValueError(f"No data found for {country}, {sex}, {age_group}")
            
            # Get last 5 years of data
            row = self.df[mask].iloc[0]
            values = []
            for year in range(2020, 2025):  # Use last 5 years
                value = row[str(year)]
                if pd.isna(value):
                    raise ValueError(f"Missing data for year {year}")
                values.append(float(value))
            
            print(f"Using historical data for prediction: {values}")
            
            # Scale values
            values = np.array(values).reshape(-1, 1)
            scaled = self.scaler.fit_transform(values)
            
            # Make predictions
            input_seq = scaled.reshape(1, 5, 1)
            predictions = []
            
            for _ in range(len(future_years)):
                pred = self.model.predict(input_seq, verbose=0)[0, 0]
                predictions.append(pred)
                
                # Update sequence for next prediction
                input_seq = np.roll(input_seq, -1, axis=1)
                input_seq[0, -1, 0] = pred
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions).flatten()
            
            # Ensure non-negative values
            predictions = np.maximum(predictions, 0)
            
            print(f"Generated predictions: {predictions.tolist()}")
            return predictions.tolist()
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return [0.0] * len(future_years)
    
    def get_available_countries(self):
        """Get list of available countries"""
        if self.df is not None:
            return sorted(self.df['country_name'].unique().tolist())
        return []
    
    def get_available_sexes(self):
        """Get list of available sex categories"""
        return self.sex_values
    
    def get_available_age_groups(self):
        """Get list of available age groups"""
        return self.age_group_values 