import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from dataset import PretrainDataset
from losses import MultiLevelNTXentLoss

def collate_pretrain(batch):
    data1_list, data2_list, masks, motifs = zip(*batch)
    return list(data1_list), list(data2_list), list(masks), list(motifs)

def pretrain_gtmocol(model, smiles_list, epochs=100, batch_size=16, lr=1e-4, temperature=0.1, alpha=1.0, beta=0.5, device='cpu', save_path='gtmocol_pretrained.pt'):
    dataset = PretrainDataset(smiles_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pretrain, num_workers=0)
    
    criterion = MultiLevelNTXentLoss(temperature=temperature, alpha=alpha, beta=beta)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    model = model.to(device)
    
    print("=" * 70)
    print("Self-Supervised Pre-training (GT-MoCoL)")
    print(f"  Molecules : {len(smiles_list)}")
    print(f"  Epochs    : {epochs}   Batch : {batch_size}   LR : {lr}")
    print(f"  α={alpha}  β={beta}  τ={temperature}")
    print("=" * 70)
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for data1_list, data2_list, motif_masks, motifs_list in loader:
            batch1 = Batch.from_data_list(data1_list).to(device)
            batch2 = Batch.from_data_list(data2_list).to(device)
            
            node1, _, z_g1 = model(batch1)
            z_m1_all, z_m1_avg = model.get_motif_projections(node1, motifs_list, batch1.batch)
            
            node2, _, z_g2 = model(batch2)
            z_m2_all, z_m2_avg = model.get_motif_projections(node2, motifs_list, batch2.batch)
            
            min_m = min(z_m1_all.size(0), z_m2_all.size(0))
            if min_m > 0:
                z_m1_all = z_m1_all[:min_m]
                z_m2_all = z_m2_all[:min_m]
            
            loss, Lg, Lm, La = criterion(z_g1, z_g2, z_m1_all, z_m2_all, z_m1_avg, z_m2_avg)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.encoder.state_dict(), save_path)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} (Lg={Lg:.3f} Lm={Lm:.3f} La={La:.3f})")
    
    print(f"\nPre-training done. Best loss: {best_loss:.4f}")
    print(f"Encoder weights saved -> {save_path}")
    return model
