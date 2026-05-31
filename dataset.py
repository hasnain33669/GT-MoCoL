import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from motif_utils import extract_motifs, get_motif_mask
from data_utils import mol_to_pyg
from augmentations import MotifAwareAugmentation

class PretrainDataset(Dataset):
    def __init__(self, smiles_list, mask_ratio=0.25):
        self.smiles = smiles_list
        self.mask_ratio = mask_ratio
        self.aug = MotifAwareAugmentation()
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        mol = None
        while mol is None:
            mol = Chem.MolFromSmiles(self.smiles[idx])
            if mol is None:
                idx = (idx + 1) % len(self.smiles)
        
        motifs = extract_motifs(mol)
        motif_mask = get_motif_mask(motifs, mol.GetNumAtoms())
        mn1 = self.aug.motif_preserving(mol, motifs, self.mask_ratio)
        data1 = mol_to_pyg(mol, mn1)
        mn2 = self.aug.motif_corrupting(mol, motifs, self.mask_ratio)
        data2 = mol_to_pyg(mol, mn2)
        return data1, data2, motif_mask, motifs

class BACEDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mol = Chem.MolFromSmiles(row['smiles'])
        motifs = extract_motifs(mol)
        motif_mask = get_motif_mask(motifs, mol.GetNumAtoms())
        data = mol_to_pyg(mol)
        data.y = torch.tensor([row['value']], dtype=torch.float)
        return data, data.y, motif_mask
