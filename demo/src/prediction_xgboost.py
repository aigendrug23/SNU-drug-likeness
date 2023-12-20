import numpy as np
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
    return prediction[0]
