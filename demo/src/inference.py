from .dataset_mtl import smilesToGeometric
from .prediction_xgboost import predict_single_vector
from .prediction_admet import get_embedding_vector, get_adme_from_vector
from .model_config import (
    get_model_admet,
    get_model_dcc,
    get_chem_space,
)
from .visualization import infer_plot, infer_dimension_reduction
import torch
import pandas as pd

encoder, predictionHead = get_model_admet()
dccModel = get_model_dcc()
chem_space = get_chem_space()


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

    # Visualization from scratch
    df = chem_space.copy()
    df = pd.concat(
        [df, pd.DataFrame({"SMILES": smiles, "isDrug": False}, index=[len(df)])]
    )
    df.loc[len(df) - 1, "0":"511"] = embedding_vector.detach().numpy()
    tsne_df = infer_dimension_reduction(
        model="TSNE",
        df=df,
        perplexity=10,
        n_iter=1000,
        metric="cosine",
        learning_rate=200,
        n_neighbors=15,
        min_dist=0.1,
    )

    # Visualization
    script, div = infer_plot(tsne_df)

    return (result_bbb, result_cyp, result_sol, result_clr), drugLikeness, (script, div)
