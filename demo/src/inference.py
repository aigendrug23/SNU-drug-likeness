from .tdc_constant import TDC
from .dataset_mtl import smilesToGeometric
from .prediction_xgboost import predict_single_vector
from .prediction_admet import get_embedding_vector, get_adme_from_vector
from .model_config import get_model_admet, get_model_dcc
import torch

encoder, predictionHead = get_model_admet()
dccModel = get_model_dcc()


def get_prediction(smiles):
    data = smilesToGeometric(smiles)
    data.batch = torch.tensor([0] * len(data.x))

    # Embedding Vector: torch.Size([1, 512])
    embedding_vector = get_embedding_vector(encoder, data)
    # ADME Prediction
    result_bbb, result_cyp, result_sol, result_clr = get_adme_from_vector(
        predictionHead, embedding_vector
    )
    # Drug Likeness Prediction
    drugLikeness = predict_single_vector(dccModel, embedding_vector.detach().numpy())

    return result_bbb, result_cyp, result_sol, result_clr, drugLikeness
