from src.inference import get_prediction
from src.model_config import get_model_admet, get_model_dcc, get_chem_space
from src.prediction_xgboost import predict_single_vector
from src.prediction_admet import get_embedding_vector, get_adme_from_vector
from src.dataset_mtl import smilesToGeometric
import torch
import pandas as pd

chem_space = get_chem_space()
dccModel = get_model_dcc()

# Visualization from scratch
df = chem_space.copy()
# embedding_vectors = df.loc[:, "0":"511"]

encoder, predictionHead = get_model_admet()

smiles = df.loc[0, "SMILES"]

ls = []
ls2 = []
# Drug Likeness Prediction
for i in df.index:
    smiles = df.loc[i, "SMILES"]
    embedding_vector = df.loc[i, "0":"511"]

    data = smilesToGeometric(smiles)
    data.batch = torch.tensor([0] * len(data.x))
    # Embedding Vector: torch.Size([1, 512])
    embedding_vector = get_embedding_vector(encoder, data)

    # ADME Prediction
    result_bbb, result_cyp, result_sol, result_clr = get_adme_from_vector(
        predictionHead, embedding_vector
    )

    drugLikeness = predict_single_vector(dccModel, embedding_vector.detach().numpy())

    if 0.3 < drugLikeness < 0.4 and 0.2 < result_bbb < 0.5:
        ls.append(
            (smiles, drugLikeness, result_bbb, result_cyp, result_sol, result_clr)
        )
    if 0.6 < drugLikeness < 0.8 and 0.2 < result_bbb < 0.5:
        ls2.append(
            (smiles, drugLikeness, result_bbb, result_cyp, result_sol, result_clr)
        )


for smi, drug, b, c, s, clr in ls:
    print(f"{smi}: {drug} for BBB: {b}, CYP: {c}, SOL: {s}, CLR: {clr}")
for smi, drug, b, c, s, clr in ls2:
    print(f"{smi}: {drug} for BBB: {b}, CYP: {c}, SOL: {s}, CLR: {clr}")
