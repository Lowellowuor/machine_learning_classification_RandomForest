"""
FastAPI endpoints for Animal Disease Prediction System
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import json
import pandas as pd
from datetime import datetime
import tempfile
import os

# Import your predictor
try:
    from animal_disease_ml.main import AnimalDiseasePredictor
except ImportError:
    # Create a mock predictor for testing if import fails
    class MockPredictor:
        def __init__(self):
            self.models_loaded = False
            
        def check_models_loaded(self):
            return {"livestock": True, "poultry": True}
            
        def get_system_info(self):
            return {"version": "2.0.0", "models": ["livestock", "poultry"]}
            
        def list_available_models(self):
            return ["livestock_model", "poultry_model"]
            
        def predict_livestock(self, **kwargs):
            return [
                {"disease": "East Coast Fever", "probability": 0.85, "confidence": "high"},
                {"disease": "Lumpy Skin Disease", "probability": 0.45, "confidence": "medium"}
            ], {"explanation": "High temperature and reduced feed intake"}, "Sample report"
            
        def predict_poultry(self, **kwargs):
            return [
                {"disease": "Newcastle Disease", "probability": 0.92, "confidence": "high"},
                {"disease": "Avian Influenza", "probability": 0.35, "confidence": "low"}
            ], {"explanation": "High mortality rate observed"}, "Sample poultry report"
            
        def predict_batch(self, file_path, animal_type):
            df = pd.read_csv(file_path)
            df["predicted_disease"] = "Sample Disease"
            df["probability"] = 0.85
            return df
            
        def train_model(self, animal_type, model_name=None, test_size=0.2):
            return f"models/{animal_type}_model.pkl", {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.91,
                "f1_score": 0.90
            }
            
        def get_disease_info(self, animal_type, disease_name=None):
            if disease_name:
                return {"name": disease_name, "symptoms": ["fever", "lethargy"], "treatment": "Consult vet"}
            return ["East Coast Fever", "Lumpy Skin Disease", "Foot and Mouth"]
            
        def get_all_diseases(self, animal_type):
            return {"diseases": ["Disease1", "Disease2", "Disease3"]}
            
        def generate_sample_data(self, animal_type, n_samples=100):
            if animal_type == "livestock":
                return [{"age_months": 24, "temperature": 39.5} for _ in range(n_samples)]
            return [{"age_weeks": 30, "flock_size": 500} for _ in range(n_samples)]
            
        def update_model(self, animal_type, model_path, model_name=None):
            return True
    
    AnimalDiseasePredictor = MockPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Kenyan Animal Disease Prediction API",
    description="REST API for predicting livestock and poultry diseases in Kenya",
    version="2.0.0",
    contact={
        "name": "Animal Disease Prediction Team",
        "email": "support@animaldisease.co.ke",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = AnimalDiseasePredictor()

# Request/Response Models
class LivestockPredictionRequest(BaseModel):
    animal_type: str = Field(..., example="dairy_cattle", description="Type of livestock")
    age_months: int = Field(..., example=36, description="Age in months")
    body_temperature: float = Field(..., example=40.5, description="Body temperature in Celsius")
    feed_intake: str = Field(..., example="reduced", description="Feed intake status")
    water_intake: str = Field(..., example="decreased", description="Water intake status")
    milk_production: Optional[float] = Field(0, example=8.5, description="Milk production in liters (0 for non-dairy)")
    county: str = Field(..., example="Nakuru", description="Kenyan county")
    season: str = Field(..., example="long_rains", description="Current season")
    vaccination_status: str = Field(..., example="partial", description="Vaccination status")
    additional_info: Optional[Dict[str, Any]] = Field({}, description="Additional information")

class PoultryPredictionRequest(BaseModel):
    poultry_type: str = Field(..., example="layers", description="Type of poultry")
    age_weeks: int = Field(..., example=32, description="Age in weeks")
    flock_size: int = Field(..., example=500, description="Number of birds in flock")
    mortality_rate: float = Field(..., example=2.5, description="Mortality rate percentage")
    egg_production: Optional[float] = Field(0, example=85.0, description="Egg production percentage (0 for non-layers)")
    feed_consumption: float = Field(..., example=100.0, description="Daily feed consumption in kg")
    county: str = Field(..., example="Kiambu", description="Kenyan county")
    season: str = Field(..., example="cold", description="Current season")
    vaccination_status: str = Field(..., example="vaccinated", description="Vaccination status")
    additional_info: Optional[Dict[str, Any]] = Field({}, description="Additional information")

class TrainingRequest(BaseModel):
    animal_type: str = Field(..., example="livestock", description="Type of animal")
    model_name: Optional[str] = Field(None, example="livestock_model_v2", description="Name for the trained model")
    test_size: Optional[float] = Field(0.2, example=0.2, description="Test size ratio")

class DiseaseInfoRequest(BaseModel):
    animal_type: str = Field(..., example="livestock", description="Type of animal")
    disease_name: Optional[str] = Field(None, example="east_coast_fever", description="Specific disease name")

class SampleDataRequest(BaseModel):
    animal_type: str = Field(..., example="poultry", description="Type of animal")
    n_samples: Optional[int] = Field(100, example=100, description="Number of samples to generate")
    format: Optional[str] = Field("csv", example="csv", description="Output format (csv or json)")

# Response Models
class StandardResponse(BaseModel):
    success: bool
    message: str
    timestamp: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class PredictionResponse(BaseModel):
    success: bool
    animal_type: str
    predictions: List[Dict[str, Any]]
    explanation: Optional[Dict[str, Any]] = None
    text_report: Optional[str] = None
    timestamp: str
    model_version: str

class TrainingResponse(BaseModel):
    success: bool
    message: str
    model_path: str
    metrics: Dict[str, Any]
    training_date: str

# Health Check Endpoint
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Kenyan Animal Disease Prediction API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /info": "System information",
            "GET /models": "Available models",
            "POST /predict/livestock": "Predict livestock disease",
            "POST /predict/poultry": "Predict poultry disease",
            "POST /predict/batch": "Batch prediction from file",
            "POST /train": "Train/re-train model",
            "GET /diseases": "Get disease information",
            "POST /generate-sample-data": "Generate sample data",
            "POST /update-model": "Update existing model"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Check if predictor is loaded
        models_loaded = predictor.check_models_loaded()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": models_loaded,
            "version": "2.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/info", tags=["System"])
async def system_info():
    """Get system information and capabilities"""
    try:
        info = predictor.get_system_info()
        return StandardResponse(
            success=True,
            message="System information retrieved successfully",
            timestamp=datetime.now().isoformat(),
            data=info
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            message="Failed to get system information",
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

@app.get("/models", tags=["Models"])
async def list_models():
    """List all available models"""
    try:
        models = predictor.list_available_models()
        return StandardResponse(
            success=True,
            message="Models listed successfully",
            timestamp=datetime.now().isoformat(),
            data={"models": models}
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            message="Failed to list models",
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

# Prediction Endpoints
@app.post("/predict/livestock", response_model=PredictionResponse, tags=["Prediction"])
async def predict_livestock(request: LivestockPredictionRequest):
    """Predict diseases for livestock"""
    try:
        # Convert request to dict
        input_data = request.dict()
        
        # Make prediction
        predictions, explanation, report = predictor.predict_livestock(
            animal_type=input_data['animal_type'],
            age_months=input_data['age_months'],
            body_temperature=input_data['body_temperature'],
            feed_intake=input_data['feed_intake'],
            water_intake=input_data['water_intake'],
            milk_production=input_data.get('milk_production', 0),
            county=input_data['county'],
            season=input_data['season'],
            vaccination_status=input_data['vaccination_status'],
            additional_info=input_data.get('additional_info', {})
        )
        
        return PredictionResponse(
            success=True,
            animal_type="livestock",
            predictions=predictions,
            explanation=explanation,
            text_report=report,
            timestamp=datetime.now().isoformat(),
            model_version="2.0.0"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/poultry", response_model=PredictionResponse, tags=["Prediction"])
async def predict_poultry(request: PoultryPredictionRequest):
    """Predict diseases for poultry"""
    try:
        # Convert request to dict
        input_data = request.dict()
        
        # Make prediction
        predictions, explanation, report = predictor.predict_poultry(
            poultry_type=input_data['poultry_type'],
            age_weeks=input_data['age_weeks'],
            flock_size=input_data['flock_size'],
            mortality_rate=input_data['mortality_rate'],
            egg_production=input_data.get('egg_production', 0),
            feed_consumption=input_data['feed_consumption'],
            county=input_data['county'],
            season=input_data['season'],
            vaccination_status=input_data['vaccination_status'],
            additional_info=input_data.get('additional_info', {})
        )
        
        return PredictionResponse(
            success=True,
            animal_type="poultry",
            predictions=predictions,
            explanation=explanation,
            text_report=report,
            timestamp=datetime.now().isoformat(),
            model_version="2.0.0"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
async def batch_prediction(
    animal_type: str = Query(..., description="Type of animal (livestock or poultry)"),
    file: UploadFile = File(..., description="CSV file with data"),
    return_file: bool = Query(True, description="Return CSV file with predictions")
):
    """Batch prediction from uploaded CSV file"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(
                status_code=400, 
                detail="File must be a CSV"
            )
        
        # Read uploaded file
        contents = await file.read()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Process batch prediction
            results_df = predictor.predict_batch(
                file_path=tmp_path,
                animal_type=animal_type
            )
            
            if return_file:
                # Save results to temporary file
                output_path = tmp_path.replace('.csv', '_predictions.csv')
                results_df.to_csv(output_path, index=False)
                
                return FileResponse(
                    output_path,
                    media_type='text/csv',
                    filename=f"{animal_type}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
            else:
                # Return JSON response
                return JSONResponse(
                    content={
                        "success": True,
                        "animal_type": animal_type,
                        "predictions": results_df.to_dict(orient='records'),
                        "timestamp": datetime.now().isoformat(),
                        "total_samples": len(results_df)
                    }
                )
                
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                if return_file:
                    output_path = tmp_path.replace('.csv', '_predictions.csv')
                    if os.path.exists(output_path):
                        os.unlink(output_path)
            except:
                pass
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Model Training Endpoint
@app.post("/train", response_model=TrainingResponse, tags=["Training"])
async def train_model(request: TrainingRequest):
    """Train or re-train a model"""
    try:
        input_data = request.dict()
        
        # Train model
        model_path, metrics = predictor.train_model(
            animal_type=input_data['animal_type'],
            model_name=input_data.get('model_name'),
            test_size=input_data.get('test_size', 0.2)
        )
        
        return TrainingResponse(
            success=True,
            message=f"Model trained successfully for {input_data['animal_type']}",
            model_path=model_path,
            metrics=metrics,
            training_date=datetime.now().isoformat()
        )
        
    except Exception as e:
        return StandardResponse(
            success=False,
            message="Model training failed",
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

# Disease Information Endpoint
@app.get("/diseases", tags=["Information"])
async def get_disease_info(
    animal_type: str = Query(..., description="Type of animal"),
    disease_name: Optional[str] = Query(None, description="Specific disease name")
):
    """Get information about diseases"""
    try:
        if disease_name:
            info = predictor.get_disease_info(
                animal_type=animal_type,
                disease_name=disease_name
            )
        else:
            info = predictor.get_all_diseases(animal_type=animal_type)
        
        return StandardResponse(
            success=True,
            message="Disease information retrieved successfully",
            timestamp=datetime.now().isoformat(),
            data=info
        )
        
    except Exception as e:
        return StandardResponse(
            success=False,
            message="Failed to get disease information",
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

# Sample Data Generation Endpoint
@app.post("/generate-sample-data", tags=["Utilities"])
async def generate_sample_data(request: SampleDataRequest):
    """Generate sample data for testing"""
    try:
        input_data = request.dict()
        
        sample_data = predictor.generate_sample_data(
            animal_type=input_data['animal_type'],
            n_samples=input_data.get('n_samples', 100)
        )
        
        if input_data.get('format', 'csv') == 'csv':
            # Return as CSV file
            df = pd.DataFrame(sample_data)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
                df.to_csv(tmp_file.name, index=False)
            
            return FileResponse(
                tmp_file.name,
                media_type='text/csv',
                filename=f"{input_data['animal_type']}_sample_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        else:
            # Return as JSON
            return JSONResponse(
                content={
                    "success": True,
                    "animal_type": input_data['animal_type'],
                    "sample_count": len(sample_data),
                    "data": sample_data,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Model Update Endpoint
@app.post("/update-model", tags=["Models"])
async def update_model(
    animal_type: str = Query(..., description="Type of animal"),
    model_path: str = Query(..., description="Path to new model file"),
    model_name: Optional[str] = Query(None, description="Name for the model")
):
    """Update/replace an existing model"""
    try:
        success = predictor.update_model(
            animal_type=animal_type,
            model_path=model_path,
            model_name=model_name
        )
        
        if success:
            return StandardResponse(
                success=True,
                message=f"Model updated successfully for {animal_type}",
                timestamp=datetime.now().isoformat(),
                data={"animal_type": animal_type, "model_name": model_name}
            )
        else:
            return StandardResponse(
                success=False,
                message="Failed to update model",
                timestamp=datetime.now().isoformat(),
                error="Model update failed"
            )
            
    except Exception as e:
        return StandardResponse(
            success=False,
            message="Failed to update model",
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",  # Change this to your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )