import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

try:
    from torch_scatter import scatter
except ImportError:
    def scatter(src, index, dim=0, dim_size=None, reduce='mean'):
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        shape = list(src.shape)
        shape[dim] = dim_size
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        if reduce == 'mean':
            count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
            out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)
            count.scatter_add_(0, index, torch.ones(index.size(0), dtype=src.dtype, device=src.device))
            count = count.clamp(min=1)
            out = out / count.unsqueeze(-1)
        elif reduce in ('add', 'sum'):
            out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)
        return out

num_atom_type = 120
num_chirality_tag = 4
num_bond_type = 5
num_bond_direction = 3

class MotifAwareAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, motif_lambda=0.5):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.motif_lambda = motif_lambda
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, motif_mask=None):
        B, N, D = x.shape
        H, Hd = self.num_heads, self.head_dim
        
        Q = self.q_proj(x).view(B, N, H, Hd)
        K = self.k_proj(x).view(B, N, H, Hd)
        V = self.v_proj(x).view(B, N, H, Hd)
        
        scores = torch.matmul(Q.permute(0, 2, 1, 3), K.permute(0, 2, 3, 1)) * self.scale
        
        if motif_mask is not None:
            scores = scores + motif_mask.unsqueeze(0).unsqueeze(0) * self.motif_lambda
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        ctx = torch.matmul(attn, V.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        ctx = ctx.reshape(B, N, D)
        return self.out_proj(ctx)

class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, motif_lambda=0.5):
        super().__init__()
        self.attn = MotifAwareAttention(hidden_dim, num_heads, dropout, motif_lambda)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, motif_mask=None):
        x = self.norm1(x + self.drop(self.attn(x, motif_mask)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x

class GTMoCoLEncoder(nn.Module):
    def __init__(self, num_layer=5, hidden_dim=300, num_heads=8, dropout=0.1, motif_lambda=0.5, pool='mean'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.motif_lambda = motif_lambda
        
        self.atom_emb = nn.Embedding(num_atom_type, hidden_dim)
        self.chiral_emb = nn.Embedding(num_chirality_tag, hidden_dim)
        self.bond_emb1 = nn.Embedding(num_bond_type, hidden_dim // 2)
        self.bond_emb2 = nn.Embedding(num_bond_direction, hidden_dim // 2)
        self.edge_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, dropout, motif_lambda)
            for _ in range(num_layer)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.pool_fn = {'mean': global_mean_pool, 'add': global_add_pool, 'max': global_max_pool}[pool]
    
    def forward(self, data, motif_mask=None):
        x = self.atom_emb(data.x[:, 0]) + self.chiral_emb(data.x[:, 1])
        
        e = torch.cat([self.bond_emb1(data.edge_attr[:, 0]), self.bond_emb2(data.edge_attr[:, 1])], dim=-1)
        e = self.edge_proj(e)
        row, col = data.edge_index
        x = x + scatter(e, col, dim=0, dim_size=x.size(0), reduce='mean')
        
        if motif_mask is not None and data.batch is not None:
            x_out = self._forward_single(x, data.batch, motif_mask)
        else:
            x_out = self._forward_batched(x, data.batch)
        
        x_out = self.norm(x_out)
        x_out = self.dropout(x_out)
        graph_repr = self.pool_fn(x_out, data.batch)
        return x_out, graph_repr
    
    def _forward_batched(self, x, batch):
        device = x.device
        unique_graphs = batch.unique()
        outputs = []
        for g in unique_graphs:
            mask = (batch == g)
            x_g = x[mask].unsqueeze(0)
            for layer in self.layers:
                x_g = layer(x_g, motif_mask=None)
            outputs.append(x_g.squeeze(0))
        return torch.cat(outputs, dim=0)
    
    def _forward_single(self, x, batch, motif_mask):
        device = x.device
        unique_graphs_indices = batch.unique()
        outputs = []
        for i, g_batch_idx in enumerate(unique_graphs_indices):
            mask_nodes_in_batch = (batch == g_batch_idx)
            x_g = x[mask_nodes_in_batch].unsqueeze(0)
            current_motif_mask = motif_mask[i].to(device)
            for layer in self.layers:
                x_g = layer(x_g, current_motif_mask)
            outputs.append(x_g.squeeze(0))
        return torch.cat(outputs, dim=0)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class GTMoCoL(nn.Module):
    def __init__(self, num_layer=5, hidden_dim=300, num_heads=8, dropout=0.1, motif_lambda=0.5, proj_dim=128, pool='mean'):
        super().__init__()
        self.encoder = GTMoCoLEncoder(num_layer, hidden_dim, num_heads, dropout, motif_lambda, pool)
        self.graph_proj = ProjectionHead(hidden_dim, hidden_dim // 2, proj_dim)
        self.motif_proj = ProjectionHead(hidden_dim, hidden_dim // 2, proj_dim)
    
    def forward(self, data, motif_mask=None):
        node_repr, graph_repr = self.encoder(data, motif_mask)
        z_G = self.graph_proj(graph_repr)
        return node_repr, graph_repr, z_G
    
    def get_motif_projections(self, node_repr, motifs_list, batch_vec):
        device = node_repr.device
        B = batch_vec.max().item() + 1
        z_all, z_avg = [], []
        
        offset = 0
        for g_idx in range(B):
            mask = (batch_vec == g_idx)
            n_nodes = mask.sum().item()
            node_g = node_repr[mask]
            motifs = motifs_list[g_idx]
            
            from motif_utils import get_motif_repr
            h_motifs = get_motif_repr(node_g, motifs)
            if h_motifs.size(0) > 0:
                z_m = self.motif_proj(h_motifs)
                z_all.append(z_m)
                z_avg.append(z_m.mean(dim=0, keepdim=True))
            else:
                z_avg.append(torch.zeros(1, self.motif_proj.net[-1].out_features, device=device))
            offset += n_nodes
        
        z_motif_all = torch.cat(z_all, dim=0) if z_all else torch.zeros(0, self.motif_proj.net[-1].out_features, device=device)
        z_motif_avg = torch.cat(z_avg, dim=0)
        return z_motif_all, z_motif_avg

class GTMoCoLForFineTuning(nn.Module):
    def __init__(self, hidden_dim=300, num_layer=5, num_heads=8, dropout=0.1, motif_lambda=0.5, num_classes=1):
        super().__init__()
        self.encoder = GTMoCoLEncoder(num_layer, hidden_dim, num_heads, dropout, motif_lambda)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, data, motif_mask=None):
        _, graph_repr = self.encoder(data, motif_mask)
        logits = self.classifier(graph_repr)
        return logits, graph_repr
    
    def load_pretrained_encoder(self, path, device='cpu'):
        state = torch.load(path, map_location=device)
        self.encoder.load_state_dict(state)
        print(f"Loaded pre-trained encoder from {path}")
