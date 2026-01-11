"""
Data validation for animal disease prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

class AnimalDataValidator:
    """Validator for animal disease data"""
    
    # Kenyan counties for validation
    KENYAN_COUNTIES = [
        'Nakuru', 'Kiambu', 'Uasin_Gishu', 'Meru', 'Kisii', 'Bungoma',
        'Kakamega', 'Kericho', 'Nyeri', 'Kilifi', 'Machakos', 'Makueni',
        'Trans_Nzoia', 'Bomet', 'Narok', 'Kajiado', 'Samburu', 'Turkana',
        'Garissa', 'Wajir', 'Mandera', 'Marsabit', 'Isiolo', 'Tana River',
        'Lamu', 'Taita Taveta', 'Kwale', 'Mombasa', 'Kilifi', 'Nyandarua',
        'Muranga', 'Kirinyaga', 'Embu', 'Tharaka Nithi', 'Laikipia', 'Nyamira',
        'Homa Bay', 'Migori', 'Kisumu', 'Siaya', 'Busia', 'Vihiga', 'Elgeyo Marakwet',
        'West Pokot', 'Samburu', 'Trans Nzoia'
    ]
    
    # Seasons in Kenya
    SEASONS = ['dry', 'short_rains', 'long_rains', 'cold']
    
    @staticmethod
    def validate_livestock_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate livestock data"""
        errors = []
        
        # Check required columns
        required_cols = ['animal_type', 'age_months', 'body_temperature']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
        
        # Validate animal types
        valid_livestock = ['dairy_cattle', 'beef_cattle', 'goats', 'sheep']
        if 'animal_type' in df.columns:
            invalid_types = df[~df['animal_type'].isin(valid_livestock)]['animal_type'].unique()
            if len(invalid_types) > 0:
                errors.append(f"Invalid animal types: {list(invalid_types)}. Valid: {valid_livestock}")
        
        # Validate numerical ranges
        if 'age_months' in df.columns:
            if df['age_months'].min() < 1:
                errors.append("Age must be at least 1 month")
            if df['age_months'].max() > 240:  # 20 years
                errors.append("Age cannot exceed 240 months (20 years)")
        
        if 'body_temperature' in df.columns:
            if df['body_temperature'].min() < 35.0:
                errors.append("Body temperature too low (minimum 35°C)")
            if df['body_temperature'].max() > 42.0:
                errors.append("Body temperature too high (maximum 42°C)")
        
        # Validate Kenyan counties
        if 'county' in df.columns:
            invalid_counties = df[~df['county'].isin(AnimalDataValidator.KENYAN_COUNTIES)]['county'].unique()
            if len(invalid_counties) > 0:
                errors.append(f"Invalid counties: {list(invalid_counties)}. Must be valid Kenyan counties")
        
        # Validate seasons
        if 'season' in df.columns:
            invalid_seasons = df[~df['season'].isin(AnimalDataValidator.SEASONS)]['season'].unique()
            if len(invalid_seasons) > 0:
                errors.append(f"Invalid seasons: {list(invalid_seasons)}. Valid: {AnimalDataValidator.SEASONS}")
        
        # Validate feed intake values
        if 'feed_intake' in df.columns:
            valid_feed = ['normal', 'increased', 'reduced', 'very_low', 'none']
            invalid_feed = df[~df['feed_intake'].isin(valid_feed)]['feed_intake'].unique()
            if len(invalid_feed) > 0:
                errors.append(f"Invalid feed intake values: {list(invalid_feed)}. Valid: {valid_feed}")
        
        # Validate water intake values
        if 'water_intake' in df.columns:
            valid_water = ['normal', 'increased', 'decreased', 'excessive', 'none']
            invalid_water = df[~df['water_intake'].isin(valid_water)]['water_intake'].unique()
            if len(invalid_water) > 0:
                errors.append(f"Invalid water intake values: {list(invalid_water)}. Valid: {valid_water}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_poultry_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate poultry data"""
        errors = []
        
        # Check required columns
        required_cols = ['poultry_type', 'age_weeks', 'flock_size']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
        
        # Validate poultry types
        valid_poultry = ['layers', 'broilers', 'local_chickens', 'turkeys', 'ducks']
        if 'poultry_type' in df.columns:
            invalid_types = df[~df['poultry_type'].isin(valid_poultry)]['poultry_type'].unique()
            if len(invalid_types) > 0:
                errors.append(f"Invalid poultry types: {list(invalid_types)}. Valid: {valid_poultry}")
        
        # Validate numerical ranges
        if 'age_weeks' in df.columns:
            if df['age_weeks'].min() < 1:
                errors.append("Age must be at least 1 week")
            if df['age_weeks'].max() > 100:  # ~2 years
                errors.append("Age cannot exceed 100 weeks")
        
        if 'flock_size' in df.columns:
            if df['flock_size'].min() < 1:
                errors.append("Flock size must be at least 1")
            if df['flock_size'].max() > 100000:
                errors.append("Flock size cannot exceed 100,000")
        
        # Validate mortality rate
        if 'mortality_rate' in df.columns:
            if df['mortality_rate'].min() < 0:
                errors.append("Mortality rate cannot be negative")
            if df['mortality_rate'].max() > 100:
                errors.append("Mortality rate cannot exceed 100%")
        
        # Validate egg production
        if 'egg_production' in df.columns:
            if df['egg_production'].min() < 0:
                errors.append("Egg production cannot be negative")
            if df['egg_production'].max() > 100:
                errors.append("Egg production cannot exceed 100%")
        
        # Validate Kenyan counties
        if 'county' in df.columns:
            invalid_counties = df[~df['county'].isin(AnimalDataValidator.KENYAN_COUNTIES)]['county'].unique()
            if len(invalid_counties) > 0:
                errors.append(f"Invalid counties: {list(invalid_counties)}")
        
        # Validate seasons
        if 'season' in df.columns:
            invalid_seasons = df[~df['season'].isin(AnimalDataValidator.SEASONS)]['season'].unique()
            if len(invalid_seasons) > 0:
                errors.append(f"Invalid seasons: {list(invalid_seasons)}. Valid: {AnimalDataValidator.SEASONS}")
        
        # Validate vaccination status
        if 'vaccination_status' in df.columns:
            valid_vaccination = ['vaccinated', 'not_vaccinated', 'partial']
            invalid_vaccination = df[~df['vaccination_status'].isin(valid_vaccination)]['vaccination_status'].unique()
            if len(invalid_vaccination) > 0:
                errors.append(f"Invalid vaccination status: {list(invalid_vaccination)}. Valid: {valid_vaccination}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def clean_data(df: pd.DataFrame, animal_type: str) -> pd.DataFrame:
        """Clean and standardize data"""
        df_clean = df.copy()
        
        # Convert column names to lowercase and strip whitespace
        df_clean.columns = [col.lower().strip().replace(' ', '_') for col in df_clean.columns]
        
        # Remove duplicate rows
        df_clean = df_clean.drop_duplicates()
        
        # Handle date columns
        date_columns = [col for col in df_clean.columns if 'date' in col or 'time' in col]
        for col in date_columns:
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            except:
                pass  # If conversion fails, leave as is
        
        # Standardize categorical values
        if animal_type == 'livestock':
            if 'animal_type' in df_clean.columns:
                df_clean['animal_type'] = df_clean['animal_type'].str.lower().str.strip().replace({
                    'cow': 'dairy_cattle',
                    'cattle': 'dairy_cattle',
                    'dairy': 'dairy_cattle',
                    'beef': 'beef_cattle',
                    'goat': 'goats',
                    'sheeps': 'sheep'
                })
            
            # Standardize feed intake values
            if 'feed_intake' in df_clean.columns:
                df_clean['feed_intake'] = df_clean['feed_intake'].str.lower().str.strip().replace({
                    'good': 'normal',
                    'ok': 'normal',
                    'poor': 'reduced',
                    'very poor': 'very_low',
                    'stopped': 'none'
                })
            
            # Standardize water intake values
            if 'water_intake' in df_clean.columns:
                df_clean['water_intake'] = df_clean['water_intake'].str.lower().str.strip().replace({
                    'good': 'normal',
                    'ok': 'normal',
                    'less': 'decreased',
                    'more': 'increased',
                    'a lot': 'excessive'
                })
        
        elif animal_type == 'poultry':
            if 'poultry_type' in df_clean.columns:
                df_clean['poultry_type'] = df_clean['poultry_type'].str.lower().str.strip().replace({
                    'chicken': 'layers',
                    'hens': 'layers',
                    'roosters': 'broilers',
                    'kienyeji': 'local_chickens',
                    'indigenous': 'local_chickens'
                })
            
            # Standardize vaccination status
            if 'vaccination_status' in df_clean.columns:
                df_clean['vaccination_status'] = df_clean['vaccination_status'].str.lower().str.strip().replace({
                    'yes': 'vaccinated',
                    'no': 'not_vaccinated',
                    'some': 'partial',
                    'few': 'partial'
                })
        
        # Standardize county names
        if 'county' in df_clean.columns:
            county_mapping = {
                'nairobi': 'Nairobi',
                'mombasa': 'Mombasa',
                'kisumu': 'Kisumu',
                'nakuru': 'Nakuru',
                'kiambu': 'Kiambu',
                'uasin gishu': 'Uasin_Gishu',
                'meru': 'Meru',
                'kisii': 'Kisii',
                'bungoma': 'Bungoma',
                'kakamega': 'Kakamega',
                'kericho': 'Kericho',
                'nyeri': 'Nyeri',
                'kilifi': 'Kilifi',
                'machakos': 'Machakos',
                'makueni': 'Makueni'
            }
            df_clean['county'] = df_clean['county'].str.title().replace(county_mapping)
        
        # Standardize season names
        if 'season' in df_clean.columns:
            season_mapping = {
                'rainy': 'long_rains',
                'wet': 'long_rains',
                'dry': 'dry',
                'cold': 'cold',
                'hot': 'dry'
            }
            df_clean['season'] = df_clean['season'].str.lower().replace(season_mapping)
        
        return df_clean
    
    @staticmethod
    def generate_schema(animal_type: str) -> Dict[str, Any]:
        """Generate data schema for animal type"""
        if animal_type == 'livestock':
            return {
                'required_columns': [
                    {'name': 'animal_type', 'type': 'categorical', 'values': ['dairy_cattle', 'beef_cattle', 'goats', 'sheep']},
                    {'name': 'age_months', 'type': 'numerical', 'range': [1, 240]},
                    {'name': 'body_temperature', 'type': 'numerical', 'range': [35.0, 42.0]},
                    {'name': 'county', 'type': 'categorical', 'values': AnimalDataValidator.KENYAN_COUNTIES}
                ],
                'optional_columns': [
                    {'name': 'feed_intake', 'type': 'categorical', 'values': ['normal', 'increased', 'reduced', 'very_low', 'none']},
                    {'name': 'water_intake', 'type': 'categorical', 'values': ['normal', 'increased', 'decreased', 'excessive', 'none']},
                    {'name': 'milk_production', 'type': 'numerical', 'range': [0, 40]},
                    {'name': 'vaccination_status', 'type': 'categorical', 'values': ['full', 'partial', 'none']},
                    {'name': 'season', 'type': 'categorical', 'values': AnimalDataValidator.SEASONS}
                ],
                'target_column': {'name': 'disease_diagnosis', 'type': 'categorical'},
                'notes': 'Milk production should be 0 for non-dairy animals'
            }
        
        elif animal_type == 'poultry':
            return {
                'required_columns': [
                    {'name': 'poultry_type', 'type': 'categorical', 'values': ['layers', 'broilers', 'local_chickens', 'turkeys', 'ducks']},
                    {'name': 'age_weeks', 'type': 'numerical', 'range': [1, 100]},
                    {'name': 'flock_size', 'type': 'numerical', 'range': [1, 100000]},
                    {'name': 'county', 'type': 'categorical', 'values': AnimalDataValidator.KENYAN_COUNTIES}
                ],
                'optional_columns': [
                    {'name': 'mortality_rate', 'type': 'numerical', 'range': [0, 100]},
                    {'name': 'egg_production', 'type': 'numerical', 'range': [0, 100]},
                    {'name': 'feed_consumption', 'type': 'numerical', 'range': [0, 1000]},
                    {'name': 'vaccination_status', 'type': 'categorical', 'values': ['vaccinated', 'not_vaccinated', 'partial']},
                    {'name': 'season', 'type': 'categorical', 'values': AnimalDataValidator.SEASONS}
                ],
                'target_column': {'name': 'disease_diagnosis', 'type': 'categorical'},
                'notes': 'Egg production should be 0 for non-laying poultry (broilers, etc.)'
            }
        
        return {}
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame, animal_type: str) -> Dict[str, Any]:
        """Check data quality metrics"""
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'data_types': {},
            'value_ranges': {},
            'unique_values': {}
        }
        
        # Check missing values
        missing = df.isnull().sum()
        quality_report['missing_values'] = missing[missing > 0].to_dict()
        quality_report['missing_percentage'] = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        # Check data types
        for col in df.columns:
            quality_report['data_types'][col] = str(df[col].dtype)
            
            # Check unique values for categorical columns
            if df[col].dtype == 'object':
                quality_report['unique_values'][col] = {
                    'count': df[col].nunique(),
                    'values': list(df[col].unique())[:10]  # First 10 unique values
                }
            
            # Check ranges for numerical columns
            if np.issubdtype(df[col].dtype, np.number):
                quality_report['value_ranges'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median())
                }
        
        # Animal-specific quality checks
        if animal_type == 'livestock':
            if 'animal_type' in df.columns:
                quality_report['animal_type_distribution'] = df['animal_type'].value_counts().to_dict()
            
            if 'body_temperature' in df.columns:
                normal_temp = df[(df['body_temperature'] >= 38.0) & (df['body_temperature'] <= 39.5)]
                quality_report['normal_temperature_percentage'] = (len(normal_temp) / len(df)) * 100
        
        elif animal_type == 'poultry':
            if 'poultry_type' in df.columns:
                quality_report['poultry_type_distribution'] = df['poultry_type'].value_counts().to_dict()
            
            if 'mortality_rate' in df.columns:
                high_mortality = df[df['mortality_rate'] > 5]
                quality_report['high_mortality_percentage'] = (len(high_mortality) / len(df)) * 100
        
        return quality_report