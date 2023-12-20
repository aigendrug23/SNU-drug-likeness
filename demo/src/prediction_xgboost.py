import numpy as np


def predict_single_vector(xgb_model, embedding_vector):
    # type checking, if not ndarray then convert it
    if type(embedding_vector) != np.ndarray:
        embedding_vector = np.array(embedding_vector)

    prediction = xgb_model.predict_proba(embedding_vector)
    return prediction[0][1]
