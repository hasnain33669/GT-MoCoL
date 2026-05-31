import torch
import yaml
from data_utils import load_bace_dataset, split_80_10_10
from model import GTMoCoLForFineTuning
from finetune import finetune_bace

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    df = load_bace_dataset('BACE.csv')
    train_df, val_df, test_df = split_80_10_10(df)
    
    model = GTMoCoLForFineTuning(
        hidden_dim=config['finetune']['hidden_dim'],
        num_layer=config['finetune']['num_layers'],
        num_heads=config['finetune']['num_heads'],
        dropout=config['finetune']['dropout'],
        motif_lambda=config['finetune']['motif_lambda'],
        num_classes=config['finetune']['num_classes']
    )
    
    model.load_pretrained_encoder('gtmocol_pretrained.pt', device)
    
    best_auc, test_acc, test_auc, test_prec, test_rec, test_f1 = finetune_bace(
        model=model,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        batch_size=config['finetune']['batch_size'],
        epochs=config['finetune']['epochs'],
        lr=config['finetune']['lr'],
        device=device,
        save_path='best_gtmocol_finetune.pt'
    )
    
    print(f"\nFinal Results:")
    print(f"Best Validation AUC: {best_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()
