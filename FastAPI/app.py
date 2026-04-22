import os
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db_config import SessionLocal, IrisRecord 

# Load environment variables from .env
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your specific HTML domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Define the incoming JSON structure
class IrisInput(BaseModel):
    sepal_len: float
    sepal_width: float
    petal_len: float
    petal_width: float

# Dependency to get the database session safely
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def intro():
    return {
        "message" : "we are FastAPI server",
        "PORT" : int(os.getenv("FASTAPI_PORT", 8000))
    }

@app.post("/predict")
def predict_route(data: IrisInput, db: Session = Depends(get_db)):
    
    # STEP 1: Check if this exact query already exists in the database
    existing_record = db.query(IrisRecord).filter(
        IrisRecord.sepal_len == data.sepal_len,
        IrisRecord.sepal_width == data.sepal_width,
        IrisRecord.petal_len == data.petal_len,
        IrisRecord.petal_width == data.petal_width
    ).first()

    # STEP 2: If found in DB, return it immediately (Skip Flask)
    if existing_record:
        return {
            "source": "MySQL Database (Cached)",
            "prediction_class": existing_record.output,
            "success": True
        }

    # STEP 3: If NOT found, send the data to your Flask API
    flask_port = os.getenv("FLASK_PORT", 5000)
    flask_api_url = f"http://127.0.0.1:{flask_port}/predict"
    
    try:
        # Convert the Pydantic model back to a dictionary and send it to Flask
        response = requests.post(flask_api_url, json=data.model_dump())
        
        # --- NEW SAFETY CHECK ---
        # If Flask crashes and sends back an HTML error page, catch it here
        if response.status_code != 200:
            return {
                "success": False, 
                "error": f"Flask Error ({response.status_code}): {response.text}"
            }
        # ------------------------

        # Parse the response from Flask (now safe to assume it is JSON)
        flask_result = response.json()
        
        return {
            "source": "Flask ML Model (Newly Predicted & Saved)",
            "prediction_class": flask_result.get("prediction_class"),
            "success": True
        }
        
    except requests.exceptions.ConnectionError:
         return {
            "success": False,
            "error": "Failed to connect to Flask API. Is the Flask server running?"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }

if __name__ == "__main__":
    # Fetch the FastAPI port from .env, default to 8000 if missing
    port = int(os.getenv("FASTAPI_PORT", 8000))
    
    # Run Uvicorn programmatically
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True)