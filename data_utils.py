import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from torch_geometric.data import Data

num_atom_type = 120
num_chirality_tag = 4
num_bond_type = 5
num_bond_direction = 3

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]

def mol_to_pyg(mol, mask_nodes=None):
    N = mol.GetNumAtoms()
    type_idx, chiral_idx = [], []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chiral_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
    
    x = torch.tensor([type_idx, chiral_idx], dtype=torch.long).t()
    
    if mask_nodes:
        for idx in mask_nodes:
            if idx < N:
                x[idx, 0] = len(ATOM_LIST)
    
    rows, cols, feats = [], [], []
    for bond in mol.GetBonds():
        s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [s, e]
        cols += [e, s]
        f = [BOND_LIST.index(bond.GetBondType()),
             BONDDIR_LIST.index(bond.GetBondDir())]
        feats += [f, f]
    
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(feats, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def load_bace_dataset(path='BACE.csv'):
    df = pd.read_csv(path)
    first = df.columns[0]
    if first.startswith(('O','C')) or 'CC' in first:
        df = pd.read_csv(path, header=None)
        df.columns = ['smiles', 'value']
    else:
        if 'class' in df.columns:
            df = df[['smiles','class']].rename(columns={'class':'value'})
        elif 'pIC50' in df.columns:
            df['value'] = (df['pIC50'] > 6.5).astype(int)
            df = df[['smiles','value']]
        else:
            lcol = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()]
            df = df[['smiles', lcol[0]]].rename(columns={lcol[0]:'value'}) if lcol \
                 else df.iloc[:, [0,-1]].rename(columns={df.columns[0]:'smiles', df.columns[-1]:'value'})
    
    df['smiles'] = df['smiles'].astype(str).str.strip().str.replace('"','')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    
    valid = [i for i,s in enumerate(df['smiles']) if Chem.MolFromSmiles(s) is not None]
    df = df.iloc[valid].reset_index(drop=True)
    df['value'] = df['value'].astype(int)
    print(f"Loaded {len(df)} valid molecules | Pos: {df['value'].sum()} Neg: {len(df)-df['value'].sum()}")
    return df

def split_80_10_10(df, seed=42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    tr, va, te = df[:int(.8*n)], df[int(.8*n):int(.9*n)], df[int(.9*n):]
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)}")
    return tr, va, te
