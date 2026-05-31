import torch
from rdkit import Chem

def extract_rings(mol):
    try:
        return [set(r) for r in Chem.GetSymmSSSR(mol)]
    except Exception:
        return []

def extract_functional_groups(mol):
    patterns = {
        'hydroxyl': '[OH]',
        'amine': '[NH2]',
        'carboxyl': 'C(=O)[OH]',
        'carbonyl': 'C=O',
        'methyl': '[CH3]',
        'methoxy': 'OC',
        'nitro': 'N(=O)=O',
        'cyano': 'C#N',
        'amide': 'C(=O)N',
        'ester': 'C(=O)OC',
        'ether': 'COC',
        'sulfide': 'CSC',
        'halogen': '[F,Cl,Br,I]',
        'benzene': 'c1ccccc1',
        'pyridine': 'n1ccccc1',
    }
    motifs = []
    for smarts in patterns.values():
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            for match in mol.GetSubstructMatches(pat):
                motifs.append(set(match))
    return motifs

def extract_motifs(mol):
    all_motifs = extract_rings(mol) + extract_functional_groups(mol)
    all_motifs = [m for m in all_motifs if len(m) > 0]
    merged = []
    for motif in all_motifs:
        placed = False
        for existing in merged:
            if motif & existing:
                existing |= motif
                placed = True
                break
        if not placed:
            merged.append(set(motif))
    return merged

def get_motif_mask(motifs, num_nodes):
    mask = torch.zeros(num_nodes, num_nodes)
    for motif in motifs:
        lst = list(motif)
        for i in lst:
            for j in lst:
                if i != j:
                    mask[i, j] = 1.0
    return mask

def get_motif_repr(node_embeddings, motifs):
    if len(motifs) == 0:
        return torch.zeros(0, node_embeddings.size(-1), device=node_embeddings.device)
    reps = []
    for motif in motifs:
        idx = list(motif)
        reps.append(node_embeddings[idx].mean(dim=0))
    return torch.stack(reps, dim=0)
