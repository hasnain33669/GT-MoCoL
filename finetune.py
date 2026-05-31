import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from dataset import BACEDataset

def finetune_bace(model, train_df, val_df, test_df, batch_size=16, epochs=100, lr=1e-4, device='cpu', save_path='best_gtmocol_finetune.pt'):
    train_ds = BACEDataset(train_df)
    val_ds = BACEDataset(val_df)
    test_ds = BACEDataset(test_df)
    
    def collate_ft(batch):
        data_list, labels, masks = zip(*batch)
        batched = Batch.from_data_list(data_list)
        return batched, torch.stack(labels), list(masks)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_ft, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_ft, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_ft, num_workers=0)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    model = model.to(device)
    
    best_auc = 0.0
    records = []
    
    print("\n" + "=" * 70)
    print("Fine-tuning GT-MoCoL on BACE")
    print("=" * 70)
    
    for epoch in range(epochs):
        t0 = time.time()
        
        model.train()
        tr_loss, tr_preds, tr_labels = 0., [], []
        for batched, labels, masks in train_loader:
            batched = batched.to(device)
            labels = labels.to(device).squeeze()
            mm = masks if masks else None
            logits, _ = model(batched, mm)
            loss = criterion(logits.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            probs = torch.sigmoid(logits.squeeze()).detach().cpu()
            tr_preds.extend((probs > .5).float().numpy())
            tr_labels.extend(labels.cpu().numpy())
        
        tr_loss /= max(len(train_loader), 1)
        tr_acc = accuracy_score(tr_labels, tr_preds)
        
        model.eval()
        vl_loss, vl_preds, vl_probs_list, vl_labels = 0., [], [], []
        with torch.no_grad():
            for batched, labels, masks in val_loader:
                batched = batched.to(device)
                labels = labels.to(device).squeeze()
                mm = masks if masks else None
                logits, _ = model(batched, mm)
                loss = criterion(logits.squeeze(), labels)
                vl_loss += loss.item()
                probs = torch.sigmoid(logits.squeeze()).cpu()
                vl_probs_list.extend(probs.numpy())
                vl_preds.extend((probs > .5).float().numpy())
                vl_labels.extend(labels.cpu().numpy())
        
        vl_loss /= max(len(val_loader), 1)
        vl_acc = accuracy_score(vl_labels, vl_preds)
        vl_auc = roc_auc_score(vl_labels, vl_probs_list) if len(set(vl_labels)) > 1 else 0.5
        
        scheduler.step()
        elapsed = time.time() - t0
        
        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(), save_path)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Time: {elapsed:.2f}s | TrLoss: {tr_loss:.4f} TrAcc: {tr_acc:.4f} | VlLoss: {vl_loss:.4f} VlAcc: {vl_acc:.4f} VlAUC: {vl_auc:.4f}")
    
    print("\n" + "=" * 70)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    te_preds, te_probs_all, te_labels = [], [], []
    with torch.no_grad():
        for batched, labels, masks in test_loader:
            batched = batched.to(device)
            mm = masks if masks else None
            logits, _ = model(batched, mm)
            probs = torch.sigmoid(logits.squeeze()).cpu()
            te_probs_all.extend(probs.numpy())
            te_preds.extend((probs > .5).float().numpy())
            te_labels.extend(labels.squeeze().numpy())
    
    te_acc = accuracy_score(te_labels, te_preds)
    te_auc = roc_auc_score(te_labels, te_probs_all) if len(set(te_labels)) > 1 else 0.5
    te_prec = precision_score(te_labels, te_preds, zero_division=0)
    te_rec = recall_score(te_labels, te_preds, zero_division=0)
    te_f1 = f1_score(te_labels, te_preds, zero_division=0)
    
    print(f"Test Accuracy : {te_acc:.4f}")
    print(f"Test AUC      : {te_auc:.4f}")
    print(f"Test Precision: {te_prec:.4f}  Recall: {te_rec:.4f}  F1: {te_f1:.4f}")
    print("=" * 70)
    
    return best_auc, te_acc, te_auc, te_prec, te_rec, te_f1
