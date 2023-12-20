from .prediction_admet import load_admet_model
from .prediction_xgboost import load_xgboost_model

model_config = {
    "admet": "models/MolCLR_[BBB, CYP3A4, Clearance, Solubility]_sc-12.13_1830.pt",
    "dcc": "models/xgboost_model.json",
}

gin, mtl = load_admet_model(model_config["admet"])
xgb_model = load_xgboost_model(model_config["dcc"])


def get_model_admet():
    return gin, mtl


def get_model_dcc():
    return xgb_model
