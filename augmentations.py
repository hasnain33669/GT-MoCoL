import random

class MotifAwareAugmentation:
    @staticmethod
    def motif_preserving(mol, motifs, mask_ratio=0.25):
        motif_atoms = {a for m in motifs for a in m}
        non_motif = [i for i in range(mol.GetNumAtoms()) if i not in motif_atoms]
        k = max(1, int(len(non_motif) * mask_ratio))
        return random.sample(non_motif, min(k, len(non_motif)))
    
    @staticmethod
    def motif_corrupting(mol, motifs, mask_ratio=0.25):
        motif_atoms = [a for m in motifs for a in m]
        k = max(1, int(len(motif_atoms) * mask_ratio))
        return random.sample(motif_atoms, min(k, len(motif_atoms))) if motif_atoms else []
