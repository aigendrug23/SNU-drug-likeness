import torch
from .tdc_constant import TDC
from joblib import load


def get_embedding_vector(encoder, data):
    return encoder(data)


def get_adme_from_vector(predictionHead, embedding_vector):
    # ADME Prediction
    predictions = predictionHead(embedding_vector).squeeze()
    result_bbb = torch.nn.Sigmoid()(predictions[TDC.allList.index(TDC.BBB)]).item()
    result_cyp = torch.nn.Sigmoid()(predictions[TDC.allList.index(TDC.CYP3A4)]).item()
    sol_scaler = load("src/scaler_solubility.bin")
    clr_scaler = load("src/scaler_clearance.bin")
    result_sol = sol_scaler.inverse_transform(
        [[predictions[TDC.allList.index(TDC.Solubility)].item()]]
    )[0][0]
    result_clr = clr_scaler.inverse_transform(
        [[predictions[TDC.allList.index(TDC.Clearance)].item()]]
    )[0][0]

    return result_bbb, result_cyp, result_sol, result_clr
