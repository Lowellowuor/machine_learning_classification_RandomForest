"""
Registry for managing multiple animal disease models
"""
from typing import Dict, Any, Optional, List
import joblib
from pathlib import Path
from datetime import datetime
import pandas as pd

from config.animal_configs import AnimalType, AnimalConfigManager
from models.livestock_model import LivestockDiseaseModel
from models.poultry_model import PoultryDiseaseModel

class ModelRegistry:
    """Registry for managing multiple animal disease models"""
    
    def __init__(self, models_dir: str = "trained_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
        self.config_manager = AnimalConfigManager()
        
    def create_model(self, animal_type: AnimalType) -> Any:
        """Create a new model instance"""
        config = self.config_manager.get_model_config(animal_type)
        
        if animal_type == AnimalType.LIVESTOCK:
            return LivestockDiseaseModel(config)
        elif animal_type == AnimalType.POULTRY:
            return PoultryDiseaseModel(config)
        else:
            raise ValueError(f"Unsupported animal type: {animal_type}")
    
    def train_and_save(self, animal_type: AnimalType, data_path: str, 
                      model_name: str = None, test_size: float = 0.2) -> Dict[str, Any]:
        """Train and save a model"""
        # Load data
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
            data = pd.read_excel(data_path)
        elif data_path.endswith('.json'):
            data = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Create model
        model = self.create_model(animal_type)
        
        # Validate data
        is_valid, errors = model.validate_input_data(data)
        if not is_valid:
            raise ValueError(f"Data validation failed: {errors}")
        
        # Check target column exists
        target_column = model.config.target_column
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Train model
        metrics = model.train(X, y, test_size=test_size)
        
        # Save model
        if not model_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{animal_type.value}_model_{timestamp}.pkl"
        
        save_path = self.models_dir / model_name
        model.save_model(str(save_path))
        
        # Register in memory
        self.loaded_models[animal_type.value] = model
        
        return {
            'model_path': str(save_path),
            'metrics': metrics,
            'animal_type': animal_type.value,
            'model_name': model_name,
            'training_date': datetime.now().isoformat(),
            'n_samples': len(data),
            'n_features': X.shape[1]
        }
    
    def load_model(self, animal_type: AnimalType, model_path: str = None) -> Any:
        """Load a trained model"""
        # Check if already loaded
        if animal_type.value in self.loaded_models:
            return self.loaded_models[animal_type.value]
        
        # Find model if path not specified
        if model_path is None:
            model_files = list(self.models_dir.glob(f"{animal_type.value}_*.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No trained model found for {animal_type.value}")
            # Get most recent model
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            model_path = str(model_files[0])
        
        # Load model data
        model_data = BaseAnimalModel.load_model(model_path)
        
        # Create model instance
        config = self.config_manager.get_model_config(animal_type)
        
        if animal_type == AnimalType.LIVESTOCK:
            model = LivestockDiseaseModel(config)
        elif animal_type == AnimalType.POULTRY:
            model = PoultryDiseaseModel(config)
        else:
            raise ValueError(f"Unsupported animal type: {animal_type}")
        
        # Restore model state
        model.model = model_data['model']
        model.feature_encoders = model_data['feature_encoders']
        model.target_encoder = model_data['target_encoder']
        model.metrics = model_data['metrics']
        model.trained_at = model_data['trained_at']
        model.model_version = model_data.get('model_version', '1.0.0')
        model.feature_columns = model_data.get('feature_columns', [])
        
        # Register in memory
        self.loaded_models[animal_type.value] = model
        
        print(f"✓ Model loaded for {animal_type.value}")
        print(f"  Path: {model_path}")
        print(f"  Accuracy: {model.metrics.get('validation_accuracy', 0):.2%}")
        print(f"  Version: {model.model_version}")
        
        return model
    
    def predict(self, animal_type: AnimalType, input_data: Dict, 
               model_path: str = None) -> Dict[str, Any]:
        """Make prediction using appropriate model"""
        # Load model
        model = self.load_model(animal_type, model_path)
        
        input_df = pd.DataFrame([input_data])
        
        # Validate input
        is_valid, errors = model.validate_input_data(input_df)
        if not is_valid:
            return {
                'error': True,
                'message': 'Input validation failed',
                'errors': errors,
                'animal_type': animal_type.value,
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Make prediction with recommendations
            if animal_type == AnimalType.LIVESTOCK:
                predictions = model.predict_with_recommendations(input_df)
            else:  # POULTRY
                predictions = model.predict_with_recommendations(input_df)
            
            return {
                'error': False,
                'animal_type': animal_type.value,
                'predictions': predictions,
                'model_version': model.model_version,
                'timestamp': datetime.now().isoformat(),
                'input_features': list(input_data.keys())
            }
        except Exception as e:
            return {
                'error': True,
                'message': f'Prediction failed: {str(e)}',
                'animal_type': animal_type.value,
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_predict(self, animal_type: AnimalType, data_file: str,
                     model_path: str = None) -> Dict[str, Any]:
        """Make batch predictions from file"""
        # Load data
        if data_file.endswith('.csv'):
            data = pd.read_csv(data_file)
        elif data_file.endswith('.json'):
            data = pd.read_json(data_file)
        else:
            return {
                'error': True,
                'message': 'Unsupported file format. Use CSV or JSON',
                'animal_type': animal_type.value
            }
        
        # Load model
        model = self.load_model(animal_type, model_path)
        
        # Validate data
        is_valid, errors = model.validate_input_data(data)
        if not is_valid:
            return {
                'error': True,
                'message': 'Data validation failed',
                'errors': errors,
                'animal_type': animal_type.value
            }
        
        try:
            # Make predictions
            if animal_type == AnimalType.LIVESTOCK:
                predictions = model.predict_with_recommendations(data)
            else:  # POULTRY
                predictions = model.predict_with_recommendations(data)
            
            # Add predictions to original data
            results_df = data.copy()
            for idx, pred in enumerate(predictions):
                if idx < len(results_df):
                    results_df.loc[idx, 'predicted_disease'] = pred['predicted_disease']
                    results_df.loc[idx, 'confidence_score'] = pred['confidence_score']
                    results_df.loc[idx, 'prediction_timestamp'] = pred['prediction_timestamp']
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.models_dir / f"predictions_{animal_type.value}_{timestamp}.csv"
            results_df.to_csv(output_file, index=False)
            
            # Generate summary
            if predictions:
                diseases = [p['predicted_disease'] for p in predictions]
                disease_counts = pd.Series(diseases).value_counts().to_dict()
                avg_confidence = sum(p['confidence_score'] for p in predictions) / len(predictions)
            else:
                disease_counts = {}
                avg_confidence = 0
            
            return {
                'error': False,
                'animal_type': animal_type.value,
                'total_predictions': len(predictions),
                'output_file': str(output_file),
                'disease_distribution': disease_counts,
                'average_confidence': avg_confidence,
                'sample_predictions': predictions[:3] if predictions else [],
                'model_version': model.model_version,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'error': True,
                'message': f'Batch prediction failed: {str(e)}',
                'animal_type': animal_type.value,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_available_models(self) -> Dict[str, List[Dict]]:
        """Get list of available trained models with metadata"""
        models_info = {}
        
        for animal_type in [AnimalType.LIVESTOCK, AnimalType.POULTRY]:
            model_files = list(self.models_dir.glob(f"{animal_type.value}_*.pkl"))
            
            models_info[animal_type.value] = []
            for model_file in model_files:
                try:
                    model_data = joblib.load(model_file)
                    models_info[animal_type.value].append({
                        'filename': model_file.name,
                        'path': str(model_file),
                        'trained_at': model_data.get('trained_at', 'Unknown'),
                        'model_version': model_data.get('model_version', '1.0.0'),
                        'accuracy': model_data.get('metrics', {}).get('validation_accuracy', 0),
                        'features': len(model_data.get('feature_columns', [])),
                        'diseases': list(model_data.get('target_encoder', {}).classes_) 
                                   if 'target_encoder' in model_data and model_data['target_encoder'] 
                                   else []
                    })
                except Exception as e:
                    models_info[animal_type.value].append({
                        'filename': model_file.name,
                        'error': f'Failed to load metadata: {str(e)}'
                    })
        
        return models_info
    
    def get_model_performance(self, animal_type: AnimalType, model_path: str = None) -> Dict[str, Any]:
        """Get detailed performance metrics for a model"""
        model = self.load_model(animal_type, model_path)
        
        return {
            'animal_type': animal_type.value,
            'model_version': model.model_version,
            'performance_metrics': model.metrics,
            'feature_importance': model.feature_importance,
            'trained_at': model.trained_at,
            'diseases_covered': list(model.target_encoder.classes_) if model.target_encoder else []
        }
    
    def delete_model(self, animal_type: AnimalType, model_name: str) -> bool:
        """Delete a trained model"""
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            return False
        
        # Remove from loaded models if present
        if animal_type.value in self.loaded_models:
            del self.loaded_models[animal_type.value]
        
        # Delete file
        model_path.unlink()
        return True