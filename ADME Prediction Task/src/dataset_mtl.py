from .tdc_data import get_mtl_data_df

import numpy as np
import torch

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def smilesToGeometric(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # add hydrogens

    N = mol.GetNumAtoms()
    M = mol.GetNumBonds()

    type_idx = []
    chirality_idx = []
    atomic_number = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)  # x[0] = node 0의 표현 = [atom, chirality]

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append(
            [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir()),
            ]
        )
        edge_feat.append(
            [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir()),
            ]
        )

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    
    # torch geometric Data의 이해
    # x: list[nodes]
    # edge_index: node 연결 정보. 양방향으로 직접 넣어준다.
    ## row: [start1, end1, start2, end2, …]
    ## col: [end1, start1, end2, start2, …]
    # 그래서 edge_feat에도 같은 feat을 두 개씩 넣어준다.

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
class MolTestDataset(Dataset):
    def __init__(self, tdcList, datasetType, scaled):
        super(Dataset, self).__init__()
        df = get_mtl_data_df(tdcList, datasetType=datasetType, scaled=scaled)

        self.smiles_data = df["Drug"].values
        self.labels = df[list(map(lambda tdc: tdc.name, tdcList))].values

    def __getitem__(self, index):
        smiles = self.smiles_data[index]
        data = smilesToGeometric(smiles)

        y = torch.tensor(self.labels[index], dtype=torch.float).view(1, -1)
        
        data.y = y
        data.smiles = smiles

#         data = Data(x=data.x, y=y, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
        return data

    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper(object):
    def __init__(
        self,
        tdcList,
        scaled,
        batch_size = 256,
        num_workers = 0,
    ):
        super(object, self).__init__()
        self.tdcList = tdcList
        self.scaled = scaled
        self.batch_size = batch_size
        self.num_workers = num_workers

    def create_data_loader(self, dataset_type):
        dataset = MolTestDataset(tdcList=self.tdcList, datasetType=dataset_type, scaled=self.scaled)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(dataset_type == "train"),  # Shuffle only for training set
            num_workers=self.num_workers,
            drop_last=False,
        )
        return loader

    """Returns train loader, valid loader, test loader
    """

    def get_data_loaders(self):
        train_loader = self.create_data_loader("train")
        valid_loader = self.create_data_loader("valid")
        test_loader = self.create_data_loader("test")

        return train_loader, valid_loader, test_loader
