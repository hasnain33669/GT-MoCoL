import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

def evaluate_with_bootstrap(model, dataset, device, n_runs=5, subsample_ratio=0.8):
    def collate_ft(batch):
        data_list, labels, masks = zip(*batch)
        batched = Batch.from_data_list(data_list)
        return batched, torch.stack(labels), list(masks)
    
    keys = ['accuracy', 'auc', 'precision', 'recall']
    met = defaultdict(list)
    
    for _ in range(n_runs):
        idx = np.random.choice(len(dataset), int(len(dataset) * subsample_ratio), replace=True)
        subset = [dataset[i] for i in idx]
        loader = DataLoader(subset, batch_size=16, shuffle=False, collate_fn=collate_ft, num_workers=0)
        
        model.eval()
        preds, probs_all, labels_all = [], [], []
        with torch.no_grad():
            for batched, labels, masks in loader:
                batched = batched.to(device)
                mm = masks if masks else None
                logits, _ = model(batched, mm)
                p = torch.sigmoid(logits.squeeze()).cpu()
                probs_all.extend(p.numpy())
                preds.extend((p > .5).float().numpy())
                labels_all.extend(labels.squeeze().numpy())
        
        met['accuracy'].append(accuracy_score(labels_all, preds))
        met['precision'].append(precision_score(labels_all, preds, zero_division=0))
        met['recall'].append(recall_score(labels_all, preds, zero_division=0))
        met['auc'].append(roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else 0.5)
    
    results = {f"{k}_mean": np.mean(met[k]) for k in keys}
    results.update({f"{k}_std": np.std(met[k]) for k in keys})
    return results
