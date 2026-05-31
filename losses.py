import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLevelNTXentLoss(nn.Module):
    def __init__(self, temperature=0.1, alpha=1.0, beta=0.5):
        super().__init__()
        self.tau = temperature
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(reduction='mean')
    
    @staticmethod
    def _nt_xent(z_i, z_j, tau):
        B = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.t()) / tau
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)
        labels = torch.cat([torch.arange(B, 2*B, device=z.device), torch.arange(0, B, device=z.device)])
        return nn.CrossEntropyLoss()(sim, labels)
    
    def graph_loss(self, z_i, z_j):
        return self._nt_xent(z_i, z_j, self.tau)
    
    def motif_loss(self, z_m_i, z_m_j):
        if z_m_i.size(0) == 0 or z_m_j.size(0) == 0:
            return torch.tensor(0.0, device=z_m_i.device)
        return self._nt_xent(z_m_i, z_m_j, self.tau)
    
    def align_loss(self, z_G, z_M):
        if z_M.size(0) == 0:
            return torch.tensor(0.0, device=z_G.device)
        return F.mse_loss(z_G, z_M)
    
    def forward(self, z_g_i, z_g_j, z_m_i, z_m_j, z_m_avg_i, z_m_avg_j):
        L_graph = self.graph_loss(z_g_i, z_g_j)
        L_motif = self.motif_loss(z_m_i, z_m_j)
        z_G_avg = (z_g_i + z_g_j) / 2
        z_MA_avg = (z_m_avg_i + z_m_avg_j) / 2
        L_align = self.align_loss(z_G_avg, z_MA_avg)
        total = L_graph + self.alpha * L_motif + self.beta * L_align
        return total, L_graph, L_motif, L_align
