from .ginet_finetune import GINet_Feat_MTL
import torch
import xgboost
import pandas as pd


model_config = {
    "admet": "models/MolCLR_[BBB, CYP3A4, Clearance, Solubility]_sc-12.13_1830.pt",
    "dcc": "models/xgboost_model.json",
    "visual": "models/tsne_perplexity20.csv",
}


def load_admet_model(model_path):
    model = GINet_Feat_MTL(
        pool="mean",
        drop_ratio=0,
        pred_layer_depth=2,
        num_tasks=4,
        pred_act="relu",
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    return model.gin, model.mtl


def load_xgboost_model(model_json_file):
    # load XGBoost model by `xgboost_model.json`

    xgb_model = xgboost.XGBClassifier()
    xgb_model.load_model(model_json_file)

    return xgb_model


def load_tsne_result(csv_path):
    df = pd.read_csv(csv_path)
    return df


gin, mtl = load_admet_model(model_config["admet"])
xgb_model = load_xgboost_model(model_config["dcc"])
tsne_result = load_tsne_result(model_config["visual"])


def get_model_admet():
    return gin, mtl


def get_model_dcc():
    return xgb_model


def get_tsne_result():
    return tsne_result
