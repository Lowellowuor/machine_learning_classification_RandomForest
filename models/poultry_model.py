"""
Poultry disease prediction model for Kenyan context
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any, Tuple, List, Union
import warnings
warnings.filterwarnings('ignore')

from models.base_animal_model import BaseAnimalModel
from config.animal_configs import AnimalType, AnimalModelConfig, AnimalConfigManager
from config.disease_mappings import DiseaseSymptomMapper

class PoultryDiseaseModel(BaseAnimalModel):
    """Poultry disease prediction model for Kenyan context"""
    
    def __init__(self, config: AnimalModelConfig):
        super().__init__(AnimalType.POULTRY, config)
        self.scaler = StandardScaler()
        self.poultry_type_encoder = LabelEncoder()
        self.feature_importance = {}
        self.model_version = "1.1.0-poultry"
        
    def validate_input_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate poultry data"""
        errors = []
        
        # Check required columns
        required_cols = self.config.required_features
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Validate data ranges
        validation_rules = self.config.validation_rules
        
        if 'age_weeks' in data.columns:
            min_age = validation_rules.get('min_age_weeks', 1)
            max_age = validation_rules.get('max_age_weeks', 100)
            if data['age_weeks'].min() < min_age:
                errors.append(f"Age too low. Minimum: {min_age} weeks")
            if data['age_weeks'].max() > max_age:
                errors.append(f"Age too high. Maximum: {max_age} weeks")
        
        if 'mortality_rate' in data.columns:
            valid_range = validation_rules.get('valid_mortality_rate', (0.0, 100.0))
            invalid = data[
                (data['mortality_rate'] < valid_range[0]) | 
                (data['mortality_rate'] > valid_range[1])
            ]
            if not invalid.empty:
                errors.append(f"Mortality rate outside valid range {valid_range}")
        
        if 'egg_production' in data.columns:
            valid_range = validation_rules.get('valid_egg_production', (0.0, 100.0))
            invalid = data[
                (data['egg_production'] < valid_range[0]) | 
                (data['egg_production'] > valid_range[1])
            ]
            if not invalid.empty:
                errors.append(f"Egg production outside valid range {valid_range}")
        
        if 'flock_size' in data.columns:
            min_size = validation_rules.get('min_flock_size', 1)
            max_size = validation_rules.get('max_flock_size', 100000)
            if data['flock_size'].min() < min_size:
                errors.append(f"Flock size too small. Minimum: {min_size}")
            if data['flock_size'].max() > max_size:
                errors.append(f"Flock size too large. Maximum: {max_size}")
        
        # Validate poultry types
        if 'poultry_type' in data.columns:
            valid_types = ['layers', 'broilers', 'local_chickens', 'turkeys', 'ducks']
            invalid_types = data[~data['poultry_type'].isin(valid_types)]['poultry_type'].unique()
            if len(invalid_types) > 0:
                errors.append(f"Invalid poultry types: {list(invalid_types)}. Valid: {valid_types}")
        
        return len(errors) == 0, errors
    
    def preprocess_data(self, data: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Preprocess poultry data"""
        df = data.copy()
        
        # Store feature columns
        self.feature_columns = self._get_feature_columns(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Calculate derived features
        df = self._calculate_derived_features(df)
        
        # Encode poultry type
        if 'poultry_type' in df.columns:
            if is_training:
                df['poultry_type_encoded'] = self.poultry_type_encoder.fit_transform(df['poultry_type'])
                self.feature_encoders['poultry_type'] = self.poultry_type_encoder
            else:
                # Handle unseen types
                unseen_mask = ~df['poultry_type'].isin(self.poultry_type_encoder.classes_)
                if unseen_mask.any():
                    df.loc[unseen_mask, 'poultry_type'] = self.poultry_type_encoder.classes_[0]
                df['poultry_type_encoded'] = self.poultry_type_encoder.transform(df['poultry_type'])
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols 
                          if col != self.config.target_column and col != 'poultry_type']
        
        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.feature_encoders[col] = le
                elif col in self.feature_encoders:
                    le = self.feature_encoders[col]
                    # Handle unseen categories
                    unseen_mask = ~df[col].isin(le.classes_)
                    if unseen_mask.any():
                        df.loc[unseen_mask, col] = le.classes_[0]
                    df[col] = le.transform(df[col].astype(str))
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols 
                         if col != self.config.target_column and 'encoded' not in col]
        
        if len(numerical_cols) > 0:
            if is_training:
                df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            else:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in poultry data"""
        # For mortality rate, use 0 if missing (assuming no deaths)
        if 'mortality_rate' in df.columns:
            df['mortality_rate'] = df['mortality_rate'].fillna(0)
        
        # For egg production in non-layers, use 0
        if 'egg_production' in df.columns:
            # Check if poultry type is layers
            if 'poultry_type' in df.columns:
                non_layers_mask = df['poultry_type'].isin(['broilers', 'local_chickens', 'turkeys', 'ducks'])
                df.loc[non_layers_mask, 'egg_production'] = df.loc[non_layers_mask, 'egg_production'].fillna(0)
            df['egg_production'] = df['egg_production'].fillna(df['egg_production'].median())
        
        # Fill other numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def _calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features for poultry"""
        # Feed Conversion Ratio (FCR) - if data available
        if all(col in df.columns for col in ['feed_consumption', 'flock_size']):
            # Avoid division by zero
            df['estimated_fcr'] = df.apply(
                lambda row: row['feed_consumption'] / (row['flock_size'] + 1) if row['flock_size'] > 0 else 0,
                axis=1
            )
        
        # Daily mortality count
        if 'mortality_rate' in df.columns and 'flock_size' in df.columns:
            df['daily_mortality_count'] = (df['mortality_rate'] / 100) * df['flock_size']
        
        # Egg production efficiency for layers
        if 'egg_production' in df.columns and 'flock_size' in df.columns:
            df['eggs_per_bird'] = df.apply(
                lambda row: row['egg_production'] / (row['flock_size'] + 1) if row['flock_size'] > 0 else 0,
                axis=1
            )
        
        # Age category
        if 'age_weeks' in df.columns:
            df['age_category'] = pd.cut(
                df['age_weeks'],
                bins=[0, 8, 20, 40, 100],
                labels=['chicks', 'growers', 'layers', 'old_layers']
            )
            # Encode age category
            le = LabelEncoder()
            df['age_category_encoded'] = le.fit_transform(df['age_category'].astype(str))
            self.feature_encoders['age_category'] = le
        
        return df
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
        """Train poultry disease model"""
        # Preprocess data
        X_processed = self.preprocess_data(X, is_training=True)
        
        # Encode target variable
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Get model parameters
        model_params = self.config.model_hyperparams.copy()
        model_type = model_params.pop('model_type', 'gradient_boosting')
        
        # Choose model based on config
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(**model_params)
        else:  # gradient_boosting default
            self.model = GradientBoostingClassifier(**model_params)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_accuracy = self.model.score(X_train, y_train)
        val_accuracy = self.model.score(X_val, y_val)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Feature importance
        self.feature_importance = self._calculate_feature_importance(X_processed)
        
        # Get classification report
        y_val_pred = self.model.predict(X_val)
        report = classification_report(y_val, y_val_pred, output_dict=True)
        
        # Store metrics
        self.metrics = {
            'train_accuracy': float(train_accuracy),
            'validation_accuracy': float(val_accuracy),
            'cross_val_mean': float(cv_scores.mean()),
            'cross_val_std': float(cv_scores.std()),
            'feature_importance': self.feature_importance,
            'classification_report': report,
            'n_samples': len(X),
            'n_features': X_processed.shape[1],
            'poultry_types': list(self.poultry_type_encoder.classes_) if hasattr(self.poultry_type_encoder, 'classes_') else []
        }
        
        self.trained_at = datetime.now().isoformat()
        
        print(f"✓ Poultry model trained successfully")
        print(f"  Validation Accuracy: {val_accuracy:.2%}")
        print(f"  Cross-Validation: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
        print(f"  Diseases learned: {len(self.target_encoder.classes_)}")
        print(f"  Poultry types: {self.metrics['poultry_types']}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame, return_confidence: bool = False) -> Union[np.ndarray, Tuple]:
        """Predict diseases for poultry"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess input data
        X_processed = self.preprocess_data(X, is_training=False)
        
        # Ensure all required columns are present
        expected_cols = self.feature_columns
        
        # Add missing columns with default values
        for col in expected_cols:
            if col not in X_processed.columns:
                X_processed[col] = 0
        
        # Reorder columns
        X_processed = X_processed[expected_cols]
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        if return_confidence:
            probabilities = self.model.predict_proba(X_processed)
            return predictions, probabilities
        
        return predictions
    
    def predict_with_recommendations(self, X: pd.DataFrame) -> List[Dict]:
        """Make predictions with poultry-specific recommendations"""
        predictions_with_details = self.predict_with_details(X)
        
        for i, result in enumerate(predictions_with_details):
            disease = result['predicted_disease']
            
            # Extract symptoms from input data
            symptoms = self._extract_symptoms(X.iloc[i] if i < len(X) else X.iloc[0])
            
            # Get disease information
            disease_info = AnimalConfigManager.get_disease_info(
                AnimalType.POULTRY, disease
            )
            
            if disease_info:
                result['disease_info'] = {
                    'common_name': disease_info.common_name,
                    'key_symptoms': disease_info.key_symptoms,
                    'high_risk_seasons': disease_info.high_risk_seasons,
                    'species_affected': disease_info.species_affected
                }
                
                result['recommendations'] = {
                    'immediate_actions': self._get_immediate_actions(disease, symptoms),
                    'treatment_guidelines': disease_info.treatment_guidelines,
                    'prevention_measures': self._get_prevention_measures(disease),
                    'biosecurity_level': self._get_biosecurity_level(disease),
                    'reporting_requirements': self._get_reporting_requirements(disease)
                }
            else:
                result['recommendations'] = {
                    'immediate_actions': 'Consult poultry veterinarian',
                    'general_advice': 'Improve biosecurity and sanitation',
                    'contact': 'County Livestock Production Office or KEPHIS'
                }
            
            # Add risk assessment
            result['risk_assessment'] = self._assess_risk(disease, X.iloc[i] if i < len(X) else X.iloc[0])
            
            # Add economic impact
            result['economic_impact'] = self._assess_economic_impact(disease, X.iloc[i] if i < len(X) else X.iloc[0])
        
        return predictions_with_details
    
    def _extract_symptoms(self, sample: pd.Series) -> List[str]:
        """Extract symptoms from sample data"""
        symptoms = []
        
        # Map features to symptoms
        symptom_mapping = {
            'mortality_rate': lambda x: 'sudden_death' if x > 10 else None,
            'egg_production': lambda x: 'reduced_egg_production' if x < 50 else None,
        }
        
        for feature, mapper in symptom_mapping.items():
            if feature in sample:
                symptom = mapper(sample[feature])
                if symptom:
                    symptoms.append(symptom)
        
        # Check for specific symptom columns
        if 'respiratory_signs' in sample and sample['respiratory_signs'] == 'yes':
            symptoms.append('respiratory_distress')
        if 'dropping_color' in sample and sample['dropping_color'] == 'green':
            symptoms.append('green_diarrhea')
        if 'dropping_color' in sample and sample['dropping_color'] == 'white':
            symptoms.append('white_diarrhea')
        
        return symptoms
    
    def _get_immediate_actions(self, disease: str, symptoms: List[str]) -> List[str]:
        """Get immediate actions for poultry disease"""
        actions = {
            'newcastle': [
                "ISOLATE SICK BIRDS IMMEDIATELY",
                "Vaccinate healthy birds if not vaccinated (consult vet)",
                "Report outbreak to veterinary authorities within 24 hours",
                "Dispose of dead birds properly (burn or deep bury with lime)",
                "Stop movement of birds, eggs, and equipment"
            ],
            'gumboro': [
                "Provide electrolyte solutions in drinking water",
                "Reduce stress factors (overcrowding, temperature fluctuations)",
                "Improve sanitation and disinfection of housing",
                "Vaccinate subsequent flocks (consult vet for schedule)",
                "Avoid mixing birds of different ages"
            ],
            'fowl_typhoid': [
                "Treat with approved antibiotics (consult vet for prescription)",
                "Improve water sanitation (chlorinate or use water sanitizers)",
                "Remove sick birds from flock immediately",
                "Disinfect housing thoroughly between batches",
                "Test and cull carrier birds"
            ],
            'avian_influenza': [
                "REPORT IMMEDIATELY to KEPHIS (0800722001)",
                "COMPLETE FARM QUARANTINE - No movement in or out",
                "Prepare for mandatory culling if confirmed",
                "Use full personal protective equipment (PPE)",
                "Document all bird movements and contacts"
            ]
        }
        
        default_actions = [
            "Isolate sick birds",
            "Improve ventilation in poultry house",
            "Provide clean water with vitamins/electrolytes",
            "Consult poultry specialist",
            "Review vaccination program"
        ]
        
        return actions.get(disease, default_actions)
    
    def _get_prevention_measures(self, disease: str) -> List[str]:
        """Get prevention measures for poultry disease"""
        measures = {
            'newcastle': [
                "Follow strict vaccination schedule (Day 1, Week 4, Week 16)",
                "Practice all-in-all-out system",
                "Control wild bird access (netting, fences)",
                "Disinfect vehicles, equipment, and footwear",
                "Source chicks from NPIP-certified hatcheries"
            ],
            'gumboro': [
                "Vaccinate parent stock for maternal antibodies",
                "Maintain strict biosecurity between houses",
                "Avoid mixing birds of different ages",
                "Proper cleaning and disinfection between batches (3-week downtime)",
                "Test drinking water quality regularly"
            ],
            'fowl_typhoid': [
                "Source chicks from Salmonella-free flocks (request test certificates)",
                "Regular water sanitization (chlorination or other methods)",
                "Rodent and insect control program",
                "Test and cull carrier birds",
                "Use competitive exclusion products (probiotics)"
            ],
            'avian_influenza': [
                "Extreme biosecurity - no contact with wild birds",
                "Report sick or dead wild birds to authorities",
                "Use farm-specific clothing and footwear",
                "Control access to farm (gates, visitor logs)",
                "Participate in government surveillance programs"
            ]
        }
        
        return measures.get(disease, [
            "Regular vaccination program (consult vet for schedule)",
            "Good nutrition with balanced feed",
            "Proper housing density (follow guidelines)",
            "Regular health monitoring and record keeping",
            "Biosecurity protocol for all visitors"
        ])
    
    def _get_biosecurity_level(self, disease: str) -> str:
        """Get required biosecurity level for disease"""
        biosecurity_levels = {
            'newcastle': 'VERY HIGH - Highly contagious, rapid spread',
            'avian_influenza': 'EXTREME - Notifiable disease, human health risk',
            'gumboro': 'HIGH - Persistent in environment, immunosuppressive',
            'fowl_typhoid': 'HIGH - Bacterial contamination risk, carrier state'
        }
        return biosecurity_levels.get(disease, 'MODERATE - Standard poultry biosecurity')
    
    def _get_reporting_requirements(self, disease: str) -> Dict[str, str]:
        """Get reporting requirements for disease"""
        reportable_diseases = {
            'avian_influenza': {
                'agency': 'KEPHIS (Kenya Plant Health Inspectorate Service)',
                'timeline': 'IMMEDIATELY (within 2 hours)',
                'contact': 'KEPHIS Hotline: 0800722001 or 0724695393',
                'penalty': 'Mandatory culling, farm quarantine, legal action'
            },
            'newcastle': {
                'agency': 'County Veterinary Office',
                'timeline': 'Within 24 hours',
                'contact': 'County Director of Veterinary Services',
                'penalty': 'Movement restrictions, mandatory vaccination'
            }
        }
        
        return reportable_diseases.get(disease, {
            'agency': 'County Livestock Office (voluntary)',
            'timeline': 'When convenient',
            'contact': 'Local veterinarian',
            'penalty': 'None'
        })
    
    def _assess_risk(self, disease: str, sample: pd.Series) -> Dict[str, Any]:
        """Assess risk level based on disease and context"""
        poultry_type = sample.get('poultry_type', 'layers')
        flock_size = sample.get('flock_size', 100)
        vaccination = sample.get('vaccination_status', 'not_vaccinated')
        
        risk_levels = {
            'avian_influenza': 'CRITICAL',
            'newcastle': 'HIGH' if vaccination == 'not_vaccinated' else 'MEDIUM',
            'gumboro': 'HIGH' if poultry_type == 'broilers' else 'MEDIUM',
            'fowl_typhoid': 'MEDIUM'
        }
        
        risk_factors = []
        if disease == 'avian_influenza':
            risk_factors.append("Notifiable disease - legal reporting required")
            risk_factors.append("Human health risk - zoonotic potential")
            risk_factors.append("Economic impact severe - trade restrictions")
        
        if disease == 'newcastle' and vaccination == 'not_vaccinated':
            risk_factors.append("Unvaccinated flock - high susceptibility")
            risk_factors.append(f"Large flock ({flock_size} birds) - rapid spread likely")
        
        if disease == 'gumboro' and poultry_type == 'broilers':
            risk_factors.append("Broilers highly susceptible to immunosuppression")
            risk_factors.append("Economic losses from reduced growth")
        
        return {
            'risk_level': risk_levels.get(disease, 'LOW'),
            'risk_factors': risk_factors,
            'containment_urgency': 'IMMEDIATE' if disease == 'avian_influenza' else 'WITHIN 24 HOURS'
        }
    
    def _assess_economic_impact(self, disease: str, sample: pd.Series) -> Dict[str, Any]:
        """Assess economic impact of disease"""
        poultry_type = sample.get('poultry_type', 'layers')
        flock_size = sample.get('flock_size', 100)
        
        # Economic impact estimates in KES
        impacts = {
            'newcastle': {
                'treatment_cost_per_bird': 50,
                'mortality_rate': '50-100% if unvaccinated',
                'egg_loss': '100% during outbreak',
                'recovery_time': '8-12 weeks',
                'vaccination_cost': 'KES 5-10 per bird'
            },
            'gumboro': {
                'treatment_cost_per_bird': 30,
                'mortality_rate': '20-30% in young birds',
                'growth_reduction': '10-15% slower growth',
                'feed_conversion': 'Increased by 0.2-0.3 points',
                'secondary_infections': 'Increased susceptibility'
            },
            'fowl_typhoid': {
                'treatment_cost_per_bird': 80,
                'mortality_rate': '10-40%',
                'egg_production_drop': '20-50%',
                'hatchability_reduction': '30-50% lower',
                'carrier_state': 'Chronic infection possible'
            },
            'avian_influenza': {
                'treatment_cost': 'N/A - mandatory culling',
                'mortality_rate': '90-100%',
                'compensation': 'KES 200-500 per bird (government)',
                'farm_quarantine': 'Minimum 3 months',
                'trade_impact': 'Export ban for 6+ months'
            }
        }
        
        impact = impacts.get(disease, {
            'treatment_cost_per_bird': 40,
            'mortality_rate': 'Variable',
            'recovery_time': '4-6 weeks'
        })
        
        # Calculate total flock impact
        if flock_size and 'treatment_cost_per_bird' in impact:
            impact['estimated_total_cost'] = flock_size * impact['treatment_cost_per_bird']
        
        # Add poultry type specific considerations
        if poultry_type == 'layers':
            impact['production_loss'] = 'Egg production drop 50-100%'
            impact['replacement_cost'] = 'KES 250-350 per point-of-lay hen'
        elif poultry_type == 'broilers':
            impact['production_loss'] = 'Weight gain reduced 10-30%'
            impact['market_weight'] = 'Reduced by 200-500g per bird'
        
        return impact