from src.inference import get_prediction
from src.model_config import get_model_admet, get_model_dcc, get_chem_space
from src.prediction_xgboost import predict_single_vector
from src.prediction_admet import get_embedding_vector
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

    # embedding_vector = embedding_vectors.iloc[i]
    embedding_vector_chem = embedding_vector.values.reshape(1, -1)
    drugLikeness_chem = predict_single_vector(dccModel, embedding_vector_chem)

    data = smilesToGeometric(smiles)
    data.batch = torch.tensor([0] * len(data.x))
    # Embedding Vector: torch.Size([1, 512])
    embedding_vector_model = get_embedding_vector(encoder, data)

    print(smiles)
    print(embedding_vector_chem[0][:5])
    print(embedding_vector_model.detach().numpy()[0][:5])

    break
    drugLikeness = predict_single_vector(dccModel, embedding_vector.detach().numpy())

    if drugLikeness < 0.4:
        ls.append((smiles, drugLikeness))
    if abs(drugLikeness_chem - drugLikeness) > 0.3:
        ls2.append((smiles, drugLikeness_chem, drugLikeness))


for smi, drug in ls:
    if drug < 0.4:
        print(smi, drug)

print(len(ls))
print(len(ls2))
