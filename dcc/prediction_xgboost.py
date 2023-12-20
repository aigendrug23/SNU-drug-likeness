import pandas as pd
import numpy as np
import sys
import xgboost

def load_xgboost_model(model_json_file):
    
    # load XGBoost model by `xgboost_model.json`
    
    xgb_model = xgboost.XGBClassifier()
    xgb_model.load_model(model_json_file)
    
    return xgb_model 


def predict_single_vector(xgb_model, embedding_vector):
    
    # type checking, if not ndarray then convert it
    if type(embedding_vector) != np.ndarray:
        embedding_vector = np.array(embedding_vector)
    
    prediction = xgb_model.predict(embedding_vector) 
    return round(prediction[0])
    
    
if __name__ == "__main__":
    xgb_model = load_xgboost_model("xgboost_model.json") # load needed only once
    print(predict_single_vector(xgb_model, [[0.5] * 512])) # use embedding vector (1 * 512) that you want to predict 
    