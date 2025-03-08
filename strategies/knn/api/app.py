from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import os
import pandas as pd
import traceback
from fractions import Fraction
from functions.base_knn import KNNClassifier
from dotenv import load_dotenv
from typing import List, Dict


# Load environment variables from .env file
load_dotenv()

app = FastAPI(debug=True)

# ------------------------------------------------------------------------------------------
# Log config
# ------------------------------------------------------------------------------------------
import logging

log_route = 'api.log'

# Configure logging
logging.basicConfig(filename=log_route, 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Log endpoint returns
def log_endpoint_return(endpoint, return_value):
    logging.info(f"Endpoint: {endpoint}, Return value: {return_value}")

# Log env variables
def log_env_vars(env_var, value):
    logging.info(f"env variable: {env_var}, env value: {value}")

#--------------------------------------------------------------
# Get method to check if the app is running
#--------------------------------------------------------------
@app.get("/")
async def root():
    return {"check":"OK", "status": "running"}

#--------------------------------------------------------------
# Establish the Data Strucuture the Api is going to expect
#--------------------------------------------------------------
class DataFrame(BaseModel):
    data: List[Dict]  # A list of dictionaries
    cols: List[str] 

#--------------------------------------------------------------
# Create a ShearingState object to ensure some variables are accesible form the whole api 
#--------------------------------------------------------------
class SharedState:
    def __init__(self):
        self.shared_dataset = str(os.getenv("DATASET_FILE"))
        self.shared_threshold = float(Fraction(os.getenv("THRESHOLD")))
        self.shared_neighbors = int(os.getenv("NEIGHBORS"))

shared_state = SharedState()

# print values picked up from .env file
print("dataset: ", shared_state.shared_dataset, "\ndataset_type: ", type(shared_state.shared_dataset))
print("threshold: ", shared_state.shared_threshold, "\nthreshold_type: ", type(shared_state.shared_threshold))
print("n_neighbors: ", shared_state.shared_neighbors, "\neighbors_type: ", type(shared_state.shared_neighbors))


# log values picked up from .env file
log_env_vars("DATASET_FILE", shared_state.shared_dataset)
log_env_vars("THRESHOLD", shared_state.shared_threshold)
log_env_vars("NEIGHBORS", shared_state.shared_neighbors)

#--------------------------------------------------------------
# Load a ClassifierModel object
#--------------------------------------------------------------

current_model = KNNClassifier(data_path=shared_state.shared_dataset, threshold=shared_state.shared_threshold, n_neighbors=shared_state.shared_neighbors)

#--------------------------------------------------------------
# POST method to make the model predict with a given input
#--------------------------------------------------------------

@app.post("/predict")
async def predict(data_frame: DataFrame):
    try:
        cols = data_frame.cols
        data = pd.DataFrame(data_frame.data)

        try:
            predictions = current_model.predict(data, cols)
        except Exception as model_error:
            # Capture full traceback of the error
            error_message = traceback.format_exc()
            print(f"Error in model.predict: {error_message}")
            return {"error": f"Model prediction error: {error_message}"}

        log_endpoint_return("/predict", "KNNClassifier")
        return {"predictions": predictions}
    
    except Exception as e:
        error_message = traceback.format_exc()
        return {'error': f"General error: {error_message}"}
    
#if __name__ == "__main__":
#    uvicorn.run(app, host='0.0.0.0', port=8081)
