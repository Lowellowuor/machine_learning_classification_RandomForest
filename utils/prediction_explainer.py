"""
Prediction explanation utilities for animal disease models
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap

class PredictionExplainer:
    """Explains model predictions in human-readable format"""
    
    @staticmethod
    def explain_livestock_prediction(prediction: Dict, input_data: Dict) -> Dict[str, Any]:
        """Explain livestock disease prediction"""
        explanation = {
            'summary': '',
            'key_factors': [],
            'confidence_interpretation': '',
            'next_steps': [],
            'warning_signs': []
        }
        
        disease = prediction.get('predicted_disease', 'unknown')
        confidence = prediction.get('confidence_score', 0)
        
        # Generate summary
        if disease == 'healthy':
            explanation['summary'] = f"The animal appears healthy with {confidence:.1%} confidence."
        else:
            explanation['summary'] = (
                f"Predicted disease: {disease.replace('_', ' ').title()} "
                f"with {confidence:.1%} confidence."
            )
        
        # Identify key factors from input data
        key_factors = []
        
        # Body temperature
        if 'body_temperature' in input_data:
            temp = input_data['body_temperature']
            if temp > 39.5:
                key_factors.append(f"High fever ({temp}°C) - indicates infection")
            elif temp < 38.0:
                key_factors.append(f"Low temperature ({temp}°C) - could indicate shock")
        
        # Feed intake
        if 'feed_intake' in input_data:
            feed = input_data['feed_intake']
            if feed in ['reduced', 'very_low', 'none']:
                key_factors.append(f"Reduced feed intake ({feed}) - common disease symptom")
        
        # Water intake
        if 'water_intake' in input_data:
            water = input_data['water_intake']
            if water == 'increased':
                key_factors.append("Increased water consumption - could indicate fever or metabolic issue")
            elif water == 'decreased':
                key_factors.append("Decreased water intake - serious concern")
        
        # Milk production for dairy
        if 'animal_type' in input_data and input_data['animal_type'] == 'dairy_cattle':
            if 'milk_production' in input_data:
                milk = input_data['milk_production']
                if milk < 10:
                    key_factors.append(f"Low milk production ({milk}L) - significant drop")
        
        # Season and location factors
        if 'season' in input_data and 'county' in input_data:
            season = input_data['season']
            county = input_data['county']
            
            if disease == 'east_coast_fever' and season in ['long_rains', 'short_rains']:
                key_factors.append(f"Rainy season ({season}) - increased tick activity")
            
            if county in ['Nakuru', 'Uasin_Gishu', 'Kericho']:
                key_factors.append(f"High-risk county ({county}) for tick-borne diseases")
        
        explanation['key_factors'] = key_factors
        
        # Confidence interpretation
        if confidence > 0.9:
            explanation['confidence_interpretation'] = "Very high confidence - strong evidence for this diagnosis"
        elif confidence > 0.7:
            explanation['confidence_interpretation'] = "High confidence - good evidence for this diagnosis"
        elif confidence > 0.5:
            explanation['confidence_interpretation'] = "Moderate confidence - consider alternative diagnoses"
        else:
            explanation['confidence_interpretation'] = "Low confidence - further investigation needed"
        
        # Next steps based on disease
        if disease == 'east_coast_fever':
            explanation['next_steps'] = [
                "1. Contact veterinarian immediately for Buparvaquone treatment",
                "2. Isolate animal to prevent spread",
                "3. Spray with acaricides to control ticks",
                "4. Monitor temperature every 6 hours"
            ]
            explanation['warning_signs'] = [
                "Rapid breathing or difficulty breathing",
                "Severe depression or inability to stand",
                "Blood in urine or feces",
                "Swelling around eyes or lymph nodes"
            ]
        
        elif disease == 'foot_and_mouth':
            explanation['next_steps'] = [
                "1. ISOLATE ANIMAL IMMEDIATELY - Highly contagious!",
                "2. Report to County Veterinary Officer within 24 hours",
                "3. Disinfect all equipment and premises",
                "4. Restrict farm access"
            ]
            explanation['warning_signs'] = [
                "Sudden lameness in multiple animals",
                "Excessive salivation or drooling",
                "Blisters on mouth, tongue, or hooves",
                "Rapid spread to other animals"
            ]
        
        elif disease == 'mastitis':
            explanation['next_steps'] = [
                "1. Milk out affected quarter completely",
                "2. Apply teat dip after milking",
                "3. Consult vet for antibiotic treatment",
                "4. Improve milking hygiene"
            ]
            explanation['warning_signs'] = [
                "Swelling or heat in udder",
                "Blood or clots in milk",
                "Fever or systemic illness",
                "Hard, painful udder quarter"
            ]
        
        elif disease == 'healthy':
            explanation['next_steps'] = [
                "1. Continue monitoring for any changes",
                "2. Maintain vaccination schedule",
                "3. Ensure proper nutrition and water",
                "4. Regular health checks"
            ]
            explanation['warning_signs'] = [
                "Any change in appetite or water intake",
                "Temperature above 39.5°C or below 38.0°C",
                "Change in behavior or activity level",
                "Any new symptoms developing"
            ]
        
        return explanation
    
    @staticmethod
    def explain_poultry_prediction(prediction: Dict, input_data: Dict) -> Dict[str, Any]:
        """Explain poultry disease prediction"""
        explanation = {
            'summary': '',
            'key_factors': [],
            'confidence_interpretation': '',
            'next_steps': [],
            'warning_signs': [],
            'economic_implications': []
        }
        
        disease = prediction.get('predicted_disease', 'unknown')
        confidence = prediction.get('confidence_score', 0)
        
        # Generate summary
        if disease == 'healthy':
            explanation['summary'] = f"The flock appears healthy with {confidence:.1%} confidence."
        else:
            explanation['summary'] = (
                f"Predicted disease: {disease.replace('_', ' ').title()} "
                f"with {confidence:.1%} confidence."
            )
        
        # Identify key factors from input data
        key_factors = []
        
        # Mortality rate
        if 'mortality_rate' in input_data:
            mortality = input_data['mortality_rate']
            if mortality > 10:
                key_factors.append(f"High mortality rate ({mortality}%) - emergency situation")
            elif mortality > 5:
                key_factors.append(f"Elevated mortality rate ({mortality}%) - serious concern")
        
        # Egg production for layers
        if 'poultry_type' in input_data and input_data['poultry_type'] == 'layers':
            if 'egg_production' in input_data:
                eggs = input_data['egg_production']
                if eggs < 50:
                    key_factors.append(f"Low egg production ({eggs}%) - significant problem")
        
        # Vaccination status
        if 'vaccination_status' in input_data:
            vaccine = input_data['vaccination_status']
            if vaccine == 'not_vaccinated':
                key_factors.append("Flock not vaccinated - high disease risk")
        
        # Flock size and density
        if 'flock_size' in input_data:
            flock_size = input_data['flock_size']
            if flock_size > 1000:
                key_factors.append(f"Large flock size ({flock_size}) - rapid disease spread risk")
        
        # Age factors
        if 'age_weeks' in input_data:
            age = input_data['age_weeks']
            if age < 4:
                key_factors.append(f"Young age ({age} weeks) - high susceptibility")
        
        explanation['key_factors'] = key_factors
        
        # Confidence interpretation
        if confidence > 0.9:
            explanation['confidence_interpretation'] = "Very high confidence - strong evidence for this diagnosis"
        elif confidence > 0.7:
            explanation['confidence_interpretation'] = "High confidence - good evidence for this diagnosis"
        elif confidence > 0.5:
            explanation['confidence_interpretation'] = "Moderate confidence - laboratory confirmation recommended"
        else:
            explanation['confidence_interpretation'] = "Low confidence - needs veterinary investigation"
        
        # Next steps based on disease
        if disease == 'newcastle':
            explanation['next_steps'] = [
                "1. ISOLATE affected birds immediately",
                "2. Vaccinate healthy birds if not vaccinated",
                "3. Report to veterinary authorities within 24 hours",
                "4. Dispose of dead birds properly (burn or bury with lime)"
            ]
            explanation['warning_signs'] = [
                "Green watery diarrhea",
                "Twisted neck or paralysis",
                "Gasping or respiratory distress",
                "Sudden high mortality"
            ]
            explanation['economic_implications'] = [
                "Mortality can reach 100% in unvaccinated flocks",
                "Egg production drops to zero during outbreak",
                "Export restrictions may apply",
                "Vaccination costs: KES 5-10 per bird"
            ]
        
        elif disease == 'gumboro':
            explanation['next_steps'] = [
                "1. Provide electrolyte solutions in drinking water",
                "2. Reduce stress factors (crowding, temperature)",
                "3. Improve sanitation and disinfection",
                "4. Vaccinate subsequent flocks properly"
            ]
            explanation['warning_signs'] = [
                "White watery diarrhea",
                "Depression and ruffled feathers",
                "Immunosuppression leading to secondary infections",
                "Uneven growth in flock"
            ]
            explanation['economic_implications'] = [
                "Reduced growth rate in broilers (10-15% slower)",
                "Increased feed conversion ratio",
                "Vaccination failure in affected flocks",
                "Treatment costs: KES 30-50 per bird"
            ]
        
        elif disease == 'fowl_typhoid':
            explanation['next_steps'] = [
                "1. Treat with approved antibiotics (consult vet)",
                "2. Improve water sanitation",
                "3. Remove sick birds from flock",
                "4. Test and cull carrier birds"
            ]
            explanation['warning_signs'] = [
                "Yellow diarrhea",
                "Reduced egg production",
                "Pale combs and wattles",
                "Sudden death in apparently healthy birds"
            ]
            explanation['economic_implications'] = [
                "Egg production drop: 20-50%",
                "Hatchability reduction: 30-50%",
                "Chronic carrier state possible",
                "Treatment costs: KES 80-120 per bird"
            ]
        
        elif disease == 'healthy':
            explanation['next_steps'] = [
                "1. Maintain biosecurity measures",
                "2. Continue vaccination program",
                "3. Monitor feed and water consumption",
                "4. Regular health checks"
            ]
            explanation['warning_signs'] = [
                "Any increase in mortality",
                "Drop in egg production",
                "Change in feed or water consumption",
                "Respiratory symptoms"
            ]
            explanation['economic_implications'] = [
                "Good flock health maximizes profits",
                "Preventive measures cheaper than treatment",
                "Regular vaccination ensures protection",
                "Good management reduces disease risk"
            ]
        
        return explanation
    
    @staticmethod
    def generate_visual_explanation(prediction: Dict, input_data: Dict, 
                                  animal_type: str, save_path: Optional[str] = None):
        """Generate visual explanation of prediction"""
        plt.figure(figsize=(12, 8))
        
        # Create subplot grid
        if animal_type == 'livestock':
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
        else:  # poultry
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            axes = axes.flatten()
        
        # Plot 1: Disease probabilities
        if 'top_disease_probabilities' in prediction:
            diseases = [d['disease'] for d in prediction['top_disease_probabilities']]
            probabilities = [d['probability'] for d in prediction['top_disease_probabilities']]
            
            axes[0].barh(diseases, probabilities, color=['#2E86AB' if d == prediction['predicted_disease'] else '#A23B72' for d in diseases])
            axes[0].set_xlabel('Probability')
            axes[0].set_title('Top Disease Probabilities')
            axes[0].set_xlim(0, 1)
        
        # Plot 2: Key factors
        explanation = PredictionExplainer.explain_livestock_prediction(prediction, input_data) if animal_type == 'livestock' \
                     else PredictionExplainer.explain_poultry_prediction(prediction, input_data)
        
        key_factors = explanation.get('key_factors', [])
        if key_factors:
            # Wrap long text
            wrapped_factors = ['\n'.join(wrap(factor, 30)) for factor in key_factors[:5]]
            y_pos = np.arange(len(wrapped_factors))
            
            axes[1].barh(y_pos, [1] * len(wrapped_factors), color='#F18F01')
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(wrapped_factors)
            axes[1].set_title('Key Contributing Factors')
            axes[1].set_xlim(0, 1.2)
        
        # Plot 3: Confidence meter
        confidence = prediction.get('confidence_score', 0)
        axes[2].bar(['Confidence'], [confidence], color='#2E86AB' if confidence > 0.7 else '#A23B72' if confidence > 0.4 else '#F18F01')
        axes[2].set_ylim(0, 1)
        axes[2].set_ylabel('Confidence Score')
        axes[2].set_title(f'Model Confidence: {confidence:.1%}')
        
        # Plot 4: Risk assessment
        if animal_type == 'livestock':
            # Use input data to assess risk
            risk_score = 0
            if 'body_temperature' in input_data and input_data['body_temperature'] > 39.5:
                risk_score += 1
            if 'feed_intake' in input_data and input_data['feed_intake'] in ['reduced', 'very_low', 'none']:
                risk_score += 1
            if 'water_intake' in input_data and input_data['water_intake'] == 'decreased':
                risk_score += 1
            
            risk_levels = ['Low', 'Medium', 'High', 'Critical']
            risk_index = min(risk_score, 3)
            
            colors = ['#2E86AB', '#F18F01', '#A23B72', '#C73E1D']
            axes[3].bar(['Risk Level'], [1], color=colors[risk_index])
            axes[3].set_title(f'Risk Assessment: {risk_levels[risk_index]}')
            axes[3].text(0, 0.5, risk_levels[risk_index], ha='center', va='center', color='white', fontsize=14, fontweight='bold')
        
        elif animal_type == 'poultry':
            # Poultry-specific risk assessment
            risk_score = 0
            if 'mortality_rate' in input_data and input_data['mortality_rate'] > 5:
                risk_score += 1
            if 'vaccination_status' in input_data and input_data['vaccination_status'] == 'not_vaccinated':
                risk_score += 1
            if 'flock_size' in input_data and input_data['flock_size'] > 1000:
                risk_score += 1
            
            risk_levels = ['Low', 'Medium', 'High', 'Critical']
            risk_index = min(risk_score, 3)
            
            colors = ['#2E86AB', '#F18F01', '#A23B72', '#C73E1D']
            axes[3].bar(['Risk Level'], [1], color=colors[risk_index])
            axes[3].set_title(f'Risk Assessment: {risk_levels[risk_index]}')
            axes[3].text(0, 0.5, risk_levels[risk_index], ha='center', va='center', color='white', fontsize=14, fontweight='bold')
            
            # Additional plot for poultry: Economic impact
            if 'economic_impact' in prediction:
                impact = prediction['economic_impact']
                if 'estimated_total_cost' in impact:
                    cost = impact['estimated_total_cost']
                    axes[4].bar(['Estimated Cost'], [cost / 1000], color='#A23B72')
                    axes[4].set_ylabel('Cost (KES, thousands)')
                    axes[4].set_title(f'Estimated Cost: KES {cost:,.0f}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def generate_text_report(prediction: Dict, input_data: Dict, 
                           animal_type: str) -> str:
        """Generate comprehensive text report"""
        if animal_type == 'livestock':
            explanation = PredictionExplainer.explain_livestock_prediction(prediction, input_data)
        else:
            explanation = PredictionExplainer.explain_poultry_prediction(prediction, input_data)
        
        report = f"""
ANIMAL DISEASE PREDICTION REPORT
{'='*50}

PREDICTION SUMMARY:
{explanation['summary']}

CONFIDENCE LEVEL:
{explanation['confidence_interpretation']}

KEY FACTORS INFLUENCING PREDICTION:
{chr(10).join(f'• {factor}' for factor in explanation['key_factors'])}

IMMEDIATE NEXT STEPS:
{chr(10).join(explanation['next_steps'])}

WARNING SIGNS TO WATCH FOR:
{chr(10).join(f'• {sign}' for sign in explanation['warning_signs'])}

MODEL INFORMATION:
• Prediction Timestamp: {prediction.get('prediction_timestamp', 'N/A')}
• Model Version: {prediction.get('model_version', 'N/A')}
• Animal Type: {animal_type.title()}

RECOMMENDATIONS:
1. Always consult with a qualified veterinarian for confirmation
2. Follow biosecurity protocols to prevent disease spread
3. Keep detailed records of symptoms and treatments
4. Report notifiable diseases to authorities as required

{'='*50}
This report is generated by the Animal Disease Prediction System.
For emergencies, contact your local veterinary officer.
"""
        
        return report
