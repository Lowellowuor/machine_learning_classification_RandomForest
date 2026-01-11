"""
Feature engineering for animal disease prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

class FeatureEngineer:
    """Engineers features for animal disease prediction"""
    
    @staticmethod
    def engineer_livestock_features(df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for livestock data"""
        df_engineered = df.copy()
        
        # 1. Age categories
        if 'age_months' in df_engineered.columns:
            df_engineered['age_category'] = pd.cut(
                df_engineered['age_months'],
                bins=[0, 12, 36, 72, 240],
                labels=['young', 'adolescent', 'adult', 'old'],
                right=False
            )
        
        # 2. Temperature status
        if 'body_temperature' in df_engineered.columns:
            df_engineered['temperature_status'] = pd.cut(
                df_engineered['body_temperature'],
                bins=[0, 38.0, 39.5, 42.0],
                labels=['normal', 'mild_fever', 'high_fever'],
                right=False
            )
        
        # 3. Production status for dairy cattle
        if all(col in df_engineered.columns for col in ['animal_type', 'milk_production']):
            df_engineered['production_status'] = df_engineered.apply(
                lambda row: 'non_dairy' if row['animal_type'] != 'dairy_cattle' else
                           'high' if row['milk_production'] > 20 else
                           'medium' if row['milk_production'] > 10 else
                           'low',
                axis=1
            )
        
        # 4. Risk score calculation
        risk_factors = 0
        
        # Age risk (very young or very old)
        if 'age_months' in df_engineered.columns:
            age_risk = ((df_engineered['age_months'] < 6) | (df_engineered['age_months'] > 120)).astype(int)
            risk_factors += age_risk
        
        # Temperature risk
        if 'body_temperature' in df_engineered.columns:
            temp_risk = (df_engineered['body_temperature'] > 39.5).astype(int)
            risk_factors += temp_risk
        
        # Feed intake risk
        if 'feed_intake' in df_engineered.columns:
            feed_risk = df_engineered['feed_intake'].isin(['reduced', 'very_low', 'none']).astype(int)
            risk_factors += feed_risk
        
        # Add risk score
        df_engineered['risk_score'] = risk_factors
        
        # 5. Season risk multiplier
        if 'season' in df_engineered.columns:
            season_risk = {
                'long_rains': 1.5,  # High tick activity
                'short_rains': 1.3,
                'dry': 1.0,
                'cold': 1.2  # Respiratory diseases
            }
            df_engineered['season_risk_multiplier'] = df_engineered['season'].map(season_risk).fillna(1.0)
        
        # 6. County risk zones
        if 'county' in df_engineered.columns:
            high_risk_counties = ['Nakuru', 'Uasin_Gishu', 'Trans_Nzoia', 'Kericho', 'Bomet']
            df_engineered['high_risk_zone'] = df_engineered['county'].isin(high_risk_counties).astype(int)
        
        return df_engineered
    
    @staticmethod
    def engineer_poultry_features(df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for poultry data"""
        df_engineered = df.copy()
        
        # 1. Age categories for poultry
        if 'age_weeks' in df_engineered.columns:
            df_engineered['age_category'] = pd.cut(
                df_engineered['age_weeks'],
                bins=[0, 4, 20, 40, 100],
                labels=['chicks', 'growers', 'layers', 'old'],
                right=False
            )
        
        # 2. Flock size categories
        if 'flock_size' in df_engineered.columns:
            df_engineered['flock_category'] = pd.cut(
                df_engineered['flock_size'],
                bins=[0, 100, 1000, 10000, 100000],
                labels=['backyard', 'small', 'medium', 'large'],
                right=False
            )
        
        # 3. Mortality severity
        if 'mortality_rate' in df_engineered.columns:
            df_engineered['mortality_severity'] = pd.cut(
                df_engineered['mortality_rate'],
                bins=[0, 1, 5, 20, 100],
                labels=['normal', 'elevated', 'high', 'severe'],
                right=False
            )
        
        # 4. Production efficiency for layers
        if all(col in df_engineered.columns for col in ['poultry_type', 'egg_production']):
            df_engineered['production_efficiency'] = df_engineered.apply(
                lambda row: 'non_layer' if row['poultry_type'] != 'layers' else
                           'excellent' if row['egg_production'] > 85 else
                           'good' if row['egg_production'] > 70 else
                           'fair' if row['egg_production'] > 50 else
                           'poor',
                axis=1
            )
        
        # 5. Risk score calculation for poultry
        risk_factors = 0
        
        # Age risk (very young more susceptible)
        if 'age_weeks' in df_engineered.columns:
            age_risk = (df_engineered['age_weeks'] < 4).astype(int)
            risk_factors += age_risk
        
        # Mortality risk
        if 'mortality_rate' in df_engineered.columns:
            mortality_risk = (df_engineered['mortality_rate'] > 5).astype(int)
            risk_factors += mortality_risk
        
        # Flock density risk (birds per square meter estimate)
        if 'flock_size' in df_engineered.columns:
            # Assuming average housing size, this is simplified
            density_risk = (df_engineered['flock_size'] > 1000).astype(int)
            risk_factors += density_risk
        
        # Vaccination risk
        if 'vaccination_status' in df_engineered.columns:
            vaccine_risk = (df_engineered['vaccination_status'] == 'not_vaccinated').astype(int)
            risk_factors += vaccine_risk
        
        # Add risk score
        df_engineered['risk_score'] = risk_factors
        
        # 6. Season disease risk
        if 'season' in df_engineered.columns:
            season_disease_risk = {
                'cold': 1.5,  # High respiratory disease risk
                'long_rains': 1.3,  # Water-borne diseases
                'short_rains': 1.2,
                'dry': 1.0
            }
            df_engineered['season_disease_risk'] = df_engineered['season'].map(season_disease_risk).fillna(1.0)
        
        # 7. Production zone
        if 'county' in df_engineered.columns:
            intensive_zones = ['Kiambu', 'Thika', 'Machakos', 'Nakuru']
            df_engineered['intensive_production_zone'] = df_engineered['county'].isin(intensive_zones).astype(int)
        
        return df_engineered
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame, animal_type: str) -> pd.DataFrame:
        """Create interaction features between variables"""
        df_interaction = df.copy()
        
        if animal_type == 'livestock':
            # Age × Temperature interaction
            if all(col in df_interaction.columns for col in ['age_months', 'body_temperature']):
                df_interaction['age_temp_interaction'] = df_interaction['age_months'] * df_interaction['body_temperature']
            
            # Season × County interaction (disease hotspots)
            if all(col in df_interaction.columns for col in ['season', 'county']):
                # Create a simple encoded interaction
                df_interaction['season_county_code'] = df_interaction['season'] + '_' + df_interaction['county']
            
            # Production × Feed interaction for dairy
            if all(col in df_interaction.columns for col in ['milk_production', 'feed_intake']):
                df_interaction['production_feed_balance'] = df_interaction.apply(
                    lambda row: 'balanced' if (row['milk_production'] > 15 and row['feed_intake'] == 'normal') else
                               'imbalanced',
                    axis=1
                )
        
        elif animal_type == 'poultry':
            # Flock size × Mortality interaction
            if all(col in df_interaction.columns for col in ['flock_size', 'mortality_rate']):
                df_interaction['flock_mortality_interaction'] = df_interaction['flock_size'] * df_interaction['mortality_rate']
            
            # Age × Production interaction for layers
            if all(col in df_interaction.columns for col in ['age_weeks', 'egg_production', 'poultry_type']):
                df_interaction['age_production_efficiency'] = df_interaction.apply(
                    lambda row: 0 if row['poultry_type'] != 'layers' else
                               row['egg_production'] / max(row['age_weeks'], 1),
                    axis=1
                )
            
            # Season × Vaccination interaction
            if all(col in df_interaction.columns for col in ['season', 'vaccination_status']):
                df_interaction['season_vaccination_risk'] = df_interaction.apply(
                    lambda row: 'high_risk' if (row['season'] == 'cold' and row['vaccination_status'] == 'not_vaccinated') else
                               'medium_risk' if (row['season'] == 'long_rains' and row['vaccination_status'] == 'not_vaccinated') else
                               'low_risk',
                    axis=1
                )
        
        return df_interaction
    
    @staticmethod
    def extract_temporal_features(df: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
        """Extract temporal features from date columns"""
        df_temporal = df.copy()
        
        if date_column and date_column in df_temporal.columns:
            try:
                dates = pd.to_datetime(df_temporal[date_column])
                
                # Extract date components
                df_temporal['year'] = dates.dt.year
                df_temporal['month'] = dates.dt.month
                df_temporal['day'] = dates.dt.day
                df_temporal['day_of_week'] = dates.dt.dayofweek
                df_temporal['quarter'] = dates.dt.quarter
                df_temporal['is_weekend'] = dates.dt.dayofweek.isin([5, 6]).astype(int)
                
                # Kenyan school terms might affect poultry demand
                school_terms = [
                    (1, 1, 3, 31),   # Jan-Mar: First term
                    (5, 1, 7, 31),   # May-Jul: Second term
                    (9, 1, 11, 30)   # Sep-Nov: Third term
                ]
                
                def get_school_term(month):
                    for i, (start_month, _, end_month, _) in enumerate(school_terms, 1):
                        if start_month <= month <= end_month:
                            return f'term_{i}'
                    return 'holiday'
                
                df_temporal['school_term'] = dates.dt.month.apply(get_school_term)
                
            except Exception as e:
                print(f"Warning: Could not extract temporal features: {e}")
        
        return df_temporal
    
    @staticmethod
    def create_aggregate_features(df: pd.DataFrame, group_column: str, 
                                 agg_columns: List[str]) -> pd.DataFrame:
        """Create aggregate features by grouping"""
        if group_column not in df.columns:
            return df
        
        df_aggregate = df.copy()
        
        # Calculate group statistics
        group_stats = df.groupby(group_column)[agg_columns].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        # Flatten column names
        group_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in group_stats.columns]
        group_stats = group_stats.rename(columns={f"{group_column}_": group_column})
        
        # Merge back to original dataframe
        df_aggregate = df_aggregate.merge(group_stats, on=group_column, how='left')
        
        return df_aggregate
