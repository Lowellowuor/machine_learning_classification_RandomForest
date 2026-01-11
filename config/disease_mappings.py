"""
Disease mappings and symptom correlations for Kenyan context
"""
from typing import Dict, List, Set
from dataclasses import dataclass

@dataclass
class SymptomMapping:
    symptom: str
    description: str
    possible_diseases: List[str]
    urgency_level: str  # low, medium, high, critical

class DiseaseSymptomMapper:
    """Maps symptoms to possible diseases for Kenyan animals"""
    
    LIVESTOCK_SYMPTOMS = {
        "fever": SymptomMapping(
            symptom="fever",
            description="Elevated body temperature (>39.5°C)",
            possible_diseases=["east_coast_fever", "foot_and_mouth", "mastitis", "lumpy_skin_disease"],
            urgency_level="high"
        ),
        "reduced_milk": SymptomMapping(
            symptom="reduced_milk",
            description="Sudden drop in milk production (>30%)",
            possible_diseases=["mastitis", "east_coast_fever", "metabolic_disorders"],
            urgency_level="medium"
        ),
        "skin_lesions": SymptomMapping(
            symptom="skin_lesions",
            description="Nodules or sores on skin",
            possible_diseases=["lumpy_skin_disease", "ringworm", "dermatitis"],
            urgency_level="medium"
        ),
        "lameness": SymptomMapping(
            symptom="lameness",
            description="Difficulty walking or standing",
            possible_diseases=["foot_and_mouth", "hoof_infections", "injuries"],
            urgency_level="high"
        ),
        "swollen_udder": SymptomMapping(
            symptom="swollen_udder",
            description="Hard, painful, or swollen mammary gland",
            possible_diseases=["mastitis", "udder_edema"],
            urgency_level="high"
        )
    }
    
    POULTRY_SYMPTOMS = {
        "green_diarrhea": SymptomMapping(
            symptom="green_diarrhea",
            description="Bright green watery droppings",
            possible_diseases=["newcastle", "fowl_cholera", "avian_influenza"],
            urgency_level="critical"
        ),
        "white_diarrhea": SymptomMapping(
            symptom="white_diarrhea",
            description="White chalky droppings",
            possible_diseases=["gumboro", "pullorum_disease"],
            urgency_level="high"
        ),
        "respiratory_distress": SymptomMapping(
            symptom="respiratory_distress",
            description="Gasping, coughing, nasal discharge",
            possible_diseases=["newcastle", "infectious_bronchitis", "avian_influenza"],
            urgency_level="high"
        ),
        "sudden_death": SymptomMapping(
            symptom="sudden_death",
            description="Unexplained rapid mortality",
            possible_diseases=["avian_influenza", "fowl_cholera", "newcastle"],
            urgency_level="critical"
        ),
        "reduced_egg_production": SymptomMapping(
            symptom="reduced_egg_production",
            description="Sharp drop in egg laying",
            possible_diseases=["newcastle", "infectious_bronchitis", "fowl_typhoid"],
            urgency_level="medium"
        )
    }
    
    @classmethod
    def get_possible_diseases(cls, symptoms: List[str], animal_type: str) -> Dict[str, List[str]]:
        """Get possible diseases based on symptoms"""
        symptom_map = cls.LIVESTOCK_SYMPTOMS if animal_type == "livestock" else cls.POULTRY_SYMPTOMS
        
        result = {}
        for symptom in symptoms:
            if symptom in symptom_map:
                mapping = symptom_map[symptom]
                result[symptom] = {
                    "possible_diseases": mapping.possible_diseases,
                    "urgency": mapping.urgency_level,
                    "description": mapping.description
                }
        
        return result
    
    @classmethod
    def calculate_urgency_score(cls, symptoms: List[str], animal_type: str) -> int:
        """Calculate urgency score based on symptoms (1-10)"""
        symptom_map = cls.LIVESTOCK_SYMPTOMS if animal_type == "livestock" else cls.POULTRY_SYMPTOMS
        
        urgency_scores = {"low": 1, "medium": 3, "high": 7, "critical": 10}
        max_score = 0
        
        for symptom in symptoms:
            if symptom in symptom_map:
                score = urgency_scores.get(symptom_map[symptom].urgency_level, 1)
                max_score = max(max_score, score)
        
        return max_score
    
    @classmethod
    def get_recommended_tests(cls, symptoms: List[str], animal_type: str) -> List[str]:
        """Get recommended diagnostic tests based on symptoms"""
        tests = []
        
        if animal_type == "livestock":
            if any(s in symptoms for s in ["fever", "skin_lesions", "swollen_lymph_nodes"]):
                tests.append("Blood smear for tick-borne diseases")
            if "lameness" in symptoms or "blisters_mouth" in symptoms:
                tests.append("FMD virus isolation or PCR")
            if "swollen_udder" in symptoms or "abnormal_milk" in symptoms:
                tests.append("Milk culture and sensitivity")
        
        elif animal_type == "poultry":
            if "green_diarrhea" in symptoms or "respiratory_distress" in symptoms:
                tests.append("Newcastle virus isolation")
            if "sudden_death" in symptoms:
                tests.append("Avian influenza PCR (REPORT TO KEPHIS)")
            if "white_diarrhea" in symptoms:
                tests.append("Bursa of Fabricius histopathology")
        
        return tests
