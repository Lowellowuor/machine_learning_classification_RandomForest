"""
Configuration management for animal disease prediction models
"""
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json

class AnimalType(Enum):
    LIVESTOCK = "livestock"  # Cattle, goats, sheep
    POULTRY = "poultry"      # Chickens, turkeys, ducks
    
class LivestockSpecies(Enum):
    DAIRY_CATTLE = "dairy_cattle"
    BEEF_CATTLE = "beef_cattle"
    GOATS = "goats"
    SHEEP = "sheep"
    
class PoultrySpecies(Enum):
    LAYERS = "layers"
    BROILERS = "broilers"
    LOCAL_CHICKENS = "local_chickens"
    TURKEYS = "turkeys"
    DUCKS = "ducks"

@dataclass
class DiseaseConfig:
    """Configuration for diseases by animal type"""
    disease_name: str
    common_name: str  # Local/Kenyan name
    species_affected: List[str]
    high_risk_seasons: List[str]
    key_symptoms: List[str]
    treatment_guidelines: Dict[str, Any]

@dataclass
class AnimalModelConfig:
    """Configuration for animal disease models"""
    animal_type: AnimalType
    required_features: List[str]
    optional_features: List[str]
    target_column: str
    preprocessing_steps: List[str]
    model_hyperparams: Dict[str, Any]
    validation_rules: Dict[str, Any]

class AnimalConfigManager:
    """Manages configurations for different animal types"""
    
    # Disease mappings for Kenya
    KENYAN_DISEASES = {
        AnimalType.LIVESTOCK: {
            "east_coast_fever": DiseaseConfig(
                disease_name="east_coast_fever",
                common_name="Nagana ya Pwani",
                species_affected=["dairy_cattle", "beef_cattle"],
                high_risk_seasons=["long_rains", "short_rains"],
                key_symptoms=["fever", "swollen_lymph_nodes", "loss_of_appetite"],
                treatment_guidelines={
                    "drug": "Buparvaquone",
                    "dosage": "2.5mg/kg IM",
                    "duration": "Single dose",
                    "cost_kes": "800-1500"
                }
            ),
            "foot_and_mouth": DiseaseConfig(
                disease_name="foot_and_mouth",
                common_name="Ugonjwa wa Mdomo na Magufuli",
                species_affected=["dairy_cattle", "beef_cattle", "goats", "sheep"],
                high_risk_seasons=["dry_season"],
                key_symptoms=["blisters_mouth", "lameness", "fever"],
                treatment_guidelines={
                    "action": "Report to authorities",
                    "isolation": "Required",
                    "vaccination": "Every 6 months"
                }
            ),
            "mastitis": DiseaseConfig(
                disease_name="mastitis",
                common_name="Ugonjwa wa Titi",
                species_affected=["dairy_cattle"],
                high_risk_seasons=["all"],
                key_symptoms=["swollen_udder", "abnormal_milk", "fever"],
                treatment_guidelines={
                    "antibiotics": ["Penicillin", "Streptomycin"],
                    "teat_dipping": "Required",
                    "milking_hygiene": "Essential"
                }
            ),
            "lumpy_skin_disease": DiseaseConfig(
                disease_name="lumpy_skin_disease",
                common_name="Ugonjwa wa Ngozi",
                species_affected=["dairy_cattle", "beef_cattle"],
                high_risk_seasons=["dry_season"],
                key_symptoms=["skin_nodules", "fever", "reduced_milk"],
                treatment_guidelines={
                    "supportive_care": "Essential",
                    "antibiotics": "For secondary infections",
                    "vaccination": "Available in some areas"
                }
            )
        },
        AnimalType.POULTRY: {
            "newcastle": DiseaseConfig(
                disease_name="newcastle",
                common_name="Ugoni wa Kuku",
                species_affected=["layers", "broilers", "local_chickens"],
                high_risk_seasons=["cold_season"],
                key_symptoms=["green_diarrhea", "nervous_signs", "respiratory_distress"],
                treatment_guidelines={
                    "vaccination": "Regular vaccination",
                    "isolation": "Immediate",
                    "sanitation": "Critical",
                    "reporting": "Mandatory in Kenya"
                }
            ),
            "gumboro": DiseaseConfig(
                disease_name="infectious_bursal_disease",
                common_name="Gumboro",
                species_affected=["broilers", "layers"],
                high_risk_seasons=["all"],
                key_symptoms=["white_diarrhea", "depression", "ruffled_feathers"],
                treatment_guidelines={
                    "vaccination": "Day-old chicks",
                    "supportive_care": "Electrolytes",
                    "biosecurity": "Enhanced",
                    "cleaning": "Thorough disinfection"
                }
            ),
            "fowl_typhoid": DiseaseConfig(
                disease_name="fowl_typhoid",
                common_name="Homa ya Kuku",
                species_affected=["layers", "broilers"],
                high_risk_seasons=["rainy_season"],
                key_symptoms=["yellow_diarrhea", "reduced_egg_production", "depression"],
                treatment_guidelines={
                    "antibiotics": ["Amoxicillin", "Tetracycline"],
                    "water_sanitation": "Essential",
                    "culling": "Severe cases",
                    "testing": "Regular flock testing"
                }
            ),
            "avian_influenza": DiseaseConfig(
                disease_name="avian_influenza",
                common_name="Homa ya Ndege",
                species_affected=["layers", "broilers", "turkeys", "ducks"],
                high_risk_seasons=["cold_season"],
                key_symptoms=["sudden_death", "swollen_head", "purple_comb"],
                treatment_guidelines={
                    "action": "REPORT IMMEDIATELY to KEPHIS",
                    "isolation": "Complete farm quarantine",
                    "culling": "Mandatory for infected flocks",
                    "compensation": "Available from government"
                }
            )
        }
    }
    
    # Model configurations
    MODEL_CONFIGS = {
        AnimalType.LIVESTOCK: AnimalModelConfig(
            animal_type=AnimalType.LIVESTOCK,
            required_features=[
                "animal_type", "age_months", "body_temperature",
                "feed_intake", "water_intake", "milk_production",
                "county", "season"
            ],
            optional_features=[
                "vaccination_status", "deworming_status", "body_condition_score",
                "rumination", "fecal_consistency", "coughing",
                "nasal_discharge", "abortion_history", "weight"
            ],
            target_column="disease_diagnosis",
            preprocessing_steps=[
                "validate_species", "encode_categorical", 
                "handle_missing", "scale_numerical"
            ],
            model_hyperparams={
                "model_type": "random_forest",
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1
            },
            validation_rules={
                "min_age": 1,
                "max_age": 240,
                "valid_temperatures": (35.0, 42.0),
                "required_counties": [
                    "Nakuru", "Kiambu", "Uasin_Gishu", "Meru", "Kisii",
                    "Bungoma", "Kakamega", "Kericho", "Nyeri", "Kilifi",
                    "Machakos", "Makueni", "Trans_Nzoia", "Bomet", "Narok"
                ]
            }
        ),
        AnimalType.POULTRY: AnimalModelConfig(
            animal_type=AnimalType.POULTRY,
            required_features=[
                "poultry_type", "age_weeks", "flock_size",
                "mortality_rate", "feed_consumption", "water_consumption",
                "egg_production", "county", "season"
            ],
            optional_features=[
                "vaccination_status", "housing_type", "ventilation",
                "dropping_color", "respiratory_signs", "neurological_signs",
                "feather_condition", "comb_color", "growth_rate"
            ],
            target_column="disease_diagnosis",
            preprocessing_steps=[
                "validate_poultry_type", "encode_categorical",
                "handle_missing", "scale_numerical", "calculate_mortality_rate"
            ],
            model_hyperparams={
                "model_type": "gradient_boosting",
                "n_estimators": 150,
                "learning_rate": 0.1,
                "max_depth": 10,
                "min_samples_split": 10,
                "min_samples_leaf": 4,
                "random_state": 42,
                "subsample": 0.8
            },
            validation_rules={
                "min_age_weeks": 1,
                "max_age_weeks": 100,
                "valid_mortality_rate": (0.0, 100.0),
                "valid_egg_production": (0.0, 100.0),
                "min_flock_size": 1,
                "max_flock_size": 100000
            }
        )
    }
    
    @classmethod
    def get_disease_info(cls, animal_type: AnimalType, disease_name: str) -> Optional[DiseaseConfig]:
        """Get disease information for specific animal type"""
        return cls.KENYAN_DISEASES.get(animal_type, {}).get(disease_name)
    
    @classmethod
    def get_model_config(cls, animal_type: AnimalType) -> AnimalModelConfig:
        """Get model configuration for animal type"""
        config = cls.MODEL_CONFIGS.get(animal_type)
        if not config:
            raise ValueError(f"No configuration found for animal type: {animal_type}")
        return config
    
    @classmethod
    def get_all_diseases(cls, animal_type: AnimalType) -> List[str]:
        """Get all diseases for animal type"""
        return list(cls.KENYAN_DISEASES.get(animal_type, {}).keys())
    
    @classmethod
    def get_species_mapping(cls, animal_type: AnimalType) -> Dict:
        """Get species mapping for animal type"""
        if animal_type == AnimalType.LIVESTOCK:
            return {e.value: e.name.replace('_', ' ').title() for e in LivestockSpecies}
        else:
            return {e.value: e.name.replace('_', ' ').title() for e in PoultrySpecies}
    
    @classmethod
    def get_county_zones(cls) -> Dict[str, List[str]]:
        """Get disease risk zones by county in Kenya"""
        return {
            "high_risk_ecf": ["Nakuru", "Uasin_Gishu", "Trans_Nzoia", "Kericho", "Bomet"],
            "high_risk_fmd": ["Kajiado", "Narok", "Samburu", "Turkana"],
            "poultry_intensive": ["Kiambu", "Thika", "Machakos", "Nakuru"],
            "dairy_zones": ["Nakuru", "Kiambu", "Uasin_Gishu", "Meru", "Nyeri"]
        }
    
    @classmethod
    def get_seasonal_risks(cls) -> Dict[str, List[str]]:
        """Get disease risks by season"""
        return {
            "long_rains": ["east_coast_fever", "foot_and_mouth", "pneumonia"],
            "short_rains": ["east_coast_fever", "mastitis"],
            "dry_season": ["foot_and_mouth", "lumpy_skin_disease", "worm_infestations"],
            "cold_season": ["pneumonia", "newcastle", "avian_influenza"]
        }