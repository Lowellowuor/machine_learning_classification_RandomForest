"""
Data loading utilities for animal disease prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import yaml
from datetime import datetime

class DataLoader:
    """Handles data loading from various sources"""
    
    @staticmethod
    def load_data(file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from various file formats"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and load accordingly
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path, **kwargs)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, **kwargs)
        elif file_path.suffix.lower() == '.json':
            return pd.read_json(file_path, **kwargs)
        elif file_path.suffix.lower() == '.parquet':
            return pd.read_parquet(file_path, **kwargs)
        elif file_path.suffix.lower() == '.feather':
            return pd.read_feather(file_path, **kwargs)
        elif file_path.suffix.lower() in ['.pkl', '.pickle']:
            return pd.read_pickle(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    @staticmethod
    def load_sample_data(animal_type: str, n_samples: int = 100) -> pd.DataFrame:
        """Generate sample data for testing"""
        np.random.seed(42)
        
        if animal_type == 'livestock':
            data = {
                'animal_type': np.random.choice(['dairy_cattle', 'beef_cattle', 'goats', 'sheep'], n_samples),
                'age_months': np.random.randint(1, 240, n_samples),
                'body_temperature': np.random.normal(38.5, 1.0, n_samples),
                'feed_intake': np.random.choice(['normal', 'reduced', 'very_low', 'increased'], n_samples, p=[0.6, 0.2, 0.1, 0.1]),
                'water_intake': np.random.choice(['normal', 'increased', 'decreased'], n_samples),
                'milk_production': np.random.uniform(0, 30, n_samples),
                'county': np.random.choice(['Nakuru', 'Kiambu', 'Uasin_Gishu', 'Meru', 'Kisii'], n_samples),
                'season': np.random.choice(['dry', 'short_rains', 'long_rains', 'cold'], n_samples),
                'vaccination_status': np.random.choice(['full', 'partial', 'none'], n_samples, p=[0.5, 0.3, 0.2]),
                'disease_diagnosis': np.random.choice(['healthy', 'east_coast_fever', 'mastitis', 'foot_and_mouth'], 
                                                    n_samples, p=[0.7, 0.15, 0.1, 0.05])
            }
            
        elif animal_type == 'poultry':
            data = {
                'poultry_type': np.random.choice(['layers', 'broilers', 'local_chickens'], n_samples),
                'age_weeks': np.random.randint(1, 100, n_samples),
                'flock_size': np.random.choice([50, 100, 500, 1000, 5000], n_samples),
                'mortality_rate': np.random.exponential(2.0, n_samples),
                'egg_production': np.random.uniform(0, 100, n_samples),
                'feed_consumption': np.random.uniform(10, 200, n_samples),
                'county': np.random.choice(['Kiambu', 'Nakuru', 'Meru', 'Kisii'], n_samples),
                'season': np.random.choice(['dry', 'short_rains', 'long_rains', 'cold'], n_samples),
                'vaccination_status': np.random.choice(['vaccinated', 'not_vaccinated', 'partial'], n_samples),
                'disease_diagnosis': np.random.choice(['healthy', 'newcastle', 'gumboro', 'fowl_typhoid'], 
                                                     n_samples, p=[0.7, 0.15, 0.1, 0.05])
            }
            
            # Adjust egg production for non-layers
            for i in range(n_samples):
                if data['poultry_type'][i] != 'layers':
                    data['egg_production'][i] = 0
        
        else:
            raise ValueError(f"Unsupported animal type: {animal_type}")
        
        return pd.DataFrame(data)
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                return json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    @staticmethod
    def save_data(df: pd.DataFrame, file_path: str, **kwargs):
        """Save data to file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix.lower() == '.csv':
            df.to_csv(file_path, index=False, **kwargs)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df.to_excel(file_path, index=False, **kwargs)
        elif file_path.suffix.lower() == '.json':
            df.to_json(file_path, orient='records', **kwargs)
        elif file_path.suffix.lower() == '.parquet':
            df.to_parquet(file_path, **kwargs)
        elif file_path.suffix.lower() == '.feather':
            df.to_feather(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    @staticmethod
    def split_data(df: pd.DataFrame, target_column: str, 
                  test_size: float = 0.2, val_size: float = 0.1, 
                  random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        from sklearn.model_selection import train_test_split
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, 
            random_state=random_state, stratify=y_train_val
        )
        
        # Combine back into dataframes
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df,
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    
    @staticmethod
    def merge_datasets(datasets: List[pd.DataFrame], 
                      merge_strategy: str = 'concat') -> pd.DataFrame:
        """Merge multiple datasets"""
        if not datasets:
            raise ValueError("No datasets provided for merging")
        
        if merge_strategy == 'concat':
            # Simple concatenation
            return pd.concat(datasets, ignore_index=True)
        
        elif merge_strategy == 'union':
            # Union of all columns, fill missing with NaN
            all_columns = set()
            for df in datasets:
                all_columns.update(df.columns)
            
            merged_data = []
            for df in datasets:
                # Add missing columns
                for col in all_columns - set(df.columns):
                    df[col] = np.nan
                # Reorder columns
                df = df[list(all_columns)]
                merged_data.append(df)
            
            return pd.concat(merged_data, ignore_index=True)
        
        else:
            raise ValueError(f"Unsupported merge strategy: {merge_strategy}")
    
    @staticmethod
    def create_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive data summary"""
        summary = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
                'duplicate_rows': df.duplicated().sum()
            },
            'data_types': df.dtypes.astype(str).to_dict(),
            'missing_values': {
                'total': df.isnull().sum().sum(),
                'by_column': df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
                'percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            },
            'numerical_stats': {},
            'categorical_stats': {}
        }
        
        # Numerical columns statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            summary['numerical_stats'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'missing': int(df[col].isnull().sum())
            }
        
        # Categorical columns statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            summary['categorical_stats'][col] = {
                'unique_values': int(df[col].nunique()),
                'top_values': value_counts.head(5).to_dict(),
                'missing': int(df[col].isnull().sum())
            }
        
        return summary
