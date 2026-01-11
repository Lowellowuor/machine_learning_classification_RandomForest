"""
Species-specific data processing
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class SpeciesProcessor:
    """Processes data based on animal species"""
    
    # Species-specific reference values
    SPECIES_REFERENCE = {
        'livestock': {
            'dairy_cattle': {
                'normal_temp_range': (38.0, 39.5),
                'normal_milk_production': (15, 30),  # liters per day
                'normal_age_range': (24, 96),  # months for peak production
                'life_expectancy': 240,  # months
                'common_breeds': ['Friesian', 'Ayrshire', 'Jersey', 'Guernsey', 'Sahiwal']
            },
            'beef_cattle': {
                'normal_temp_range': (38.0, 39.5),
                'normal_weight_gain': (0.8, 1.2),  # kg per day
                'normal_age_range': (12, 60),  # months for slaughter
                'life_expectancy': 180,  # months
                'common_breeds': ['Borana', 'Zebu', 'Santa Gertrudis', 'Charolais']
            },
            'goats': {
                'normal_temp_range': (38.5, 40.0),
                'normal_milk_production': (1, 3),  # liters per day
                'normal_age_range': (12, 84),  # months
                'life_expectancy': 144,  # months
                'common_breeds': ['Small East African', 'Galla', 'Toggenburg', 'Saanen']
            },
            'sheep': {
                'normal_temp_range': (38.5, 40.0),
                'normal_weight': (30, 70),  # kg
                'normal_age_range': (12, 72),  # months
                'life_expectancy': 120,  # months
                'common_breeds': ['Red Maasai', 'Dorper', 'Merino', 'Hampshire']
            }
        },
        'poultry': {
            'layers': {
                'normal_egg_production': (70, 95),  # percentage
                'peak_production_age': (24, 52),  # weeks
                'feed_conversion': 2.0,  # kg feed per kg eggs
                'common_breeds': ['Lohmann Brown', 'Hy-Line', 'ISA Brown', 'Bovans']
            },
            'broilers': {
                'normal_weight_gain': (50, 70),  # grams per day
                'slaughter_age': (6, 8),  # weeks
                'feed_conversion': 1.6,  # kg feed per kg weight gain
                'common_breeds': ['Cobb', 'Ross', 'Hubbard', 'Arbor Acres']
            },
            'local_chickens': {
                'normal_egg_production': (30, 60),  # percentage
                'mature_weight': (1.5, 2.5),  # kg
                'foraging_ability': 'high',
                'disease_resistance': 'high'
            },
            'turkeys': {
                'normal_weight': (8, 15),  # kg
                'slaughter_age': (16, 24),  # weeks
                'feed_conversion': 2.5,
                'common_breeds': ['Broad Breasted White', 'Bronze', 'Bourbon Red']
            },
            'ducks': {
                'normal_egg_production': (60, 85),  # percentage
                'slaughter_age': (8, 12),  # weeks
                'feed_conversion': 2.2,
                'common_breeds': ['Pekin', 'Muscovy', 'Khaki Campbell']
            }
        }
    }
    
    @staticmethod
    def normalize_by_species(df: pd.DataFrame, animal_type: str) -> pd.DataFrame:
        """Normalize values based on species reference ranges"""
        df_normalized = df.copy()
        
        if animal_type == 'livestock' and 'animal_type' in df_normalized.columns:
            for species, ref_values in SpeciesProcessor.SPECIES_REFERENCE['livestock'].items():
                species_mask = df_normalized['animal_type'] == species
                
                # Normalize temperature
                if 'body_temperature' in df_normalized.columns:
                    temp_min, temp_max = ref_values['normal_temp_range']
                    df_normalized.loc[species_mask, 'temp_normalized'] = (
                        (df_normalized.loc[species_mask, 'body_temperature'] - temp_min) / 
                        (temp_max - temp_min)
                    )
                
                # Normalize age
                if 'age_months' in df_normalized.columns:
                    life_expectancy = ref_values['life_expectancy']
                    df_normalized.loc[species_mask, 'age_normalized'] = (
                        df_normalized.loc[species_mask, 'age_months'] / life_expectancy
                    )
                
                # For dairy cattle, normalize milk production
                if species == 'dairy_cattle' and 'milk_production' in df_normalized.columns:
                    prod_min, prod_max = ref_values['normal_milk_production']
                    df_normalized.loc[species_mask, 'milk_production_normalized'] = (
                        df_normalized.loc[species_mask, 'milk_production'] / prod_max
                    )
        
        elif animal_type == 'poultry' and 'poultry_type' in df_normalized.columns:
            for species, ref_values in SpeciesProcessor.SPECIES_REFERENCE['poultry'].items():
                species_mask = df_normalized['poultry_type'] == species
                
                # Normalize age for poultry
                if 'age_weeks' in df_normalized.columns:
                    if species == 'broilers':
                        slaughter_age = ref_values['slaughter_age'][1]
                        df_normalized.loc[species_mask, 'age_normalized'] = (
                            df_normalized.loc[species_mask, 'age_weeks'] / slaughter_age
                        )
                    elif species == 'layers':
                        peak_age = ref_values['peak_production_age'][1]
                        df_normalized.loc[species_mask, 'age_normalized'] = (
                            df_normalized.loc[species_mask, 'age_weeks'] / peak_age
                        )
                
                # Normalize egg production for layers
                if species == 'layers' and 'egg_production' in df_normalized.columns:
                    prod_min, prod_max = ref_values['normal_egg_production']
                    df_normalized.loc[species_mask, 'egg_production_normalized'] = (
                        (df_normalized.loc[species_mask, 'egg_production'] - prod_min) / 
                        (prod_max - prod_min)
                    )
        
        return df_normalized
    
    @staticmethod
    def calculate_health_indices(df: pd.DataFrame, animal_type: str) -> pd.DataFrame:
        """Calculate health indices based on species"""
        df_indices = df.copy()
        
        if animal_type == 'livestock' and 'animal_type' in df_indices.columns:
            # Initialize health index
            df_indices['health_index'] = 100  # Start with perfect health
            
            # Deduct points for abnormalities
            
            # Temperature deductions
            if 'body_temperature' in df_indices.columns:
                def temp_deduction(temp, species):
                    if species == 'dairy_cattle' or species == 'beef_cattle':
                        normal_min, normal_max = 38.0, 39.5
                    else:  # goats, sheep
                        normal_min, normal_max = 38.5, 40.0
                    
                    if temp < normal_min:
                        return 20 + (normal_min - temp) * 10
                    elif temp > normal_max:
                        return 20 + (temp - normal_max) * 10
                    return 0
                
                df_indices['temp_deduction'] = df_indices.apply(
                    lambda row: temp_deduction(row['body_temperature'], row['animal_type']), 
                    axis=1
                )
                df_indices['health_index'] -= df_indices['temp_deduction']
            
            # Feed intake deductions
            if 'feed_intake' in df_indices.columns:
                feed_deductions = {
                    'normal': 0,
                    'increased': 5,
                    'decreased': 15,
                    'reduced': 25,
                    'very_low': 40,
                    'none': 60
                }
                df_indices['feed_deduction'] = df_indices['feed_intake'].map(feed_deductions).fillna(0)
                df_indices['health_index'] -= df_indices['feed_deduction']
            
            # Age-based adjustments
            if 'age_months' in df_indices.columns:
                def age_adjustment(age, species):
                    ref = SpeciesProcessor.SPECIES_REFERENCE['livestock'].get(species, {})
                    life_expectancy = ref.get('life_expectancy', 120)
                    age_ratio = age / life_expectancy
                    
                    if age_ratio > 0.8:  # Old age
                        return -20
                    elif age_ratio < 0.1:  # Very young
                        return -10
                    return 0
                
                df_indices['age_adjustment'] = df_indices.apply(
                    lambda row: age_adjustment(row['age_months'], row['animal_type']), 
                    axis=1
                )
                df_indices['health_index'] += df_indices['age_adjustment']
        
        elif animal_type == 'poultry' and 'poultry_type' in df_indices.columns:
            # Initialize health index
            df_indices['health_index'] = 100
            
            # Mortality rate deductions
            if 'mortality_rate' in df_indices.columns:
                def mortality_deduction(rate, poultry_type):
                    if poultry_type in ['broilers', 'layers']:
                        if rate > 10:
                            return 50
                        elif rate > 5:
                            return 30
                        elif rate > 2:
                            return 15
                    else:  # local chickens more resilient
                        if rate > 20:
                            return 50
                        elif rate > 10:
                            return 30
                        elif rate > 5:
                            return 15
                    return 0
                
                df_indices['mortality_deduction'] = df_indices.apply(
                    lambda row: mortality_deduction(row['mortality_rate'], row['poultry_type']), 
                    axis=1
                )
                df_indices['health_index'] -= df_indices['mortality_deduction']
            
            # Egg production deductions for layers
            if all(col in df_indices.columns for col in ['egg_production', 'poultry_type']):
                def egg_production_deduction(production, poultry_type):
                    if poultry_type == 'layers':
                        if production < 50:
                            return 40
                        elif production < 70:
                            return 20
                        elif production < 85:
                            return 5
                    return 0
                
                df_indices['production_deduction'] = df_indices.apply(
                    lambda row: egg_production_deduction(row['egg_production'], row['poultry_type']), 
                    axis=1
                )
                df_indices['health_index'] -= df_indices['production_deduction']
            
            # Vaccination bonus
            if 'vaccination_status' in df_indices.columns:
                vaccination_bonus = {
                    'vaccinated': 10,
                    'partial': 5,
                    'not_vaccinated': 0
                }
                df_indices['vaccination_bonus'] = df_indices['vaccination_status'].map(vaccination_bonus).fillna(0)
                df_indices['health_index'] += df_indices['vaccination_bonus']
        
        # Ensure health index stays within reasonable bounds
        df_indices['health_index'] = df_indices['health_index'].clip(0, 100)
        
        # Categorize health status
        df_indices['health_status'] = pd.cut(
            df_indices['health_index'],
            bins=[0, 40, 70, 90, 100],
            labels=['critical', 'poor', 'fair', 'good'],
            right=False
        )
        
        return df_indices
    
    @staticmethod
    def impute_missing_by_species(df: pd.DataFrame, animal_type: str) -> pd.DataFrame:
        """Impute missing values based on species patterns"""
        df_imputed = df.copy()
        
        if animal_type == 'livestock' and 'animal_type' in df_imputed.columns:
            for species in df_imputed['animal_type'].unique():
                species_mask = df_imputed['animal_type'] == species
                ref_values = SpeciesProcessor.SPECIES_REFERENCE['livestock'].get(species, {})
                
                # Impute body temperature
                if 'body_temperature' in df_imputed.columns:
                    if species in ref_values:
                        normal_min, normal_max = ref_values['normal_temp_range']
                        default_temp = (normal_min + normal_max) / 2
                        missing_temp = species_mask & df_imputed['body_temperature'].isna()
                        df_imputed.loc[missing_temp, 'body_temperature'] = default_temp
                
                # Impute age with median of species
                if 'age_months' in df_imputed.columns:
                    species_median_age = df_imputed.loc[species_mask, 'age_months'].median()
                    if pd.isna(species_median_age) and species in ref_values:
                        # Use reference normal age range midpoint
                        age_range = ref_values.get('normal_age_range', (12, 60))
                        species_median_age = sum(age_range) / 2
                    missing_age = species_mask & df_imputed['age_months'].isna()
                    df_imputed.loc[missing_age, 'age_months'] = species_median_age
        
        elif animal_type == 'poultry' and 'poultry_type' in df_imputed.columns:
            for species in df_imputed['poultry_type'].unique():
                species_mask = df_imputed['poultry_type'] == species
                ref_values = SpeciesProcessor.SPECIES_REFERENCE['poultry'].get(species, {})
                
                # Impute egg production
                if 'egg_production' in df_imputed.columns:
                    if species == 'layers' and species in ref_values:
                        prod_min, prod_max = ref_values['normal_egg_production']
                        default_production = (prod_min + prod_max) / 2
                    else:
                        default_production = 0  # Non-layers don't lay eggs
                    missing_production = species_mask & df_imputed['egg_production'].isna()
                    df_imputed.loc[missing_production, 'egg_production'] = default_production
                
                # Impute age in weeks
                if 'age_weeks' in df_imputed.columns:
                    species_median_age = df_imputed.loc[species_mask, 'age_weeks'].median()
                    if pd.isna(species_median_age) and species in ref_values:
                        if species == 'broilers':
                            age_range = ref_values.get('slaughter_age', (6, 8))
                        elif species == 'layers':
                            age_range = ref_values.get('peak_production_age', (24, 52))
                        else:
                            age_range = (12, 52)  # Default
                        species_median_age = sum(age_range) / 2
                    missing_age = species_mask & df_imputed['age_weeks'].isna()
                    df_imputed.loc[missing_age, 'age_weeks'] = species_median_age
        
        return df_imputed
