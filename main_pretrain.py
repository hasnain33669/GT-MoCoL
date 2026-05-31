import torch
import yaml
from data_utils import load_bace_dataset
from model import GTMoCoL
from pretrain import pretrain_gtmocol

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    df = load_bace_dataset('BACE.csv')
    smiles_list = df['smiles'].tolist()
    
    model = GTMoCoL(
        num_layer=config['pretrain']['num_layers'],
        hidden_dim=config['pretrain']['hidden_dim'],
        num_heads=config['pretrain']['num_heads'],
        dropout=config['pretrain']['dropout'],
        motif_lambda=config['pretrain']['motif_lambda'],
        proj_dim=config['pretrain']['proj_dim'],
        pool=config['pretrain']['pool']
    )
    
    model = pretrain_gtmocol(
        model=model,
        smiles_list=smiles_list,
        epochs=config['pretrain']['epochs'],
        batch_size=config['pretrain']['batch_size'],
        lr=config['pretrain']['lr'],
        temperature=config['pretrain']['temperature'],
        alpha=config['pretrain']['alpha'],
        beta=config['pretrain']['beta'],
        device=device,
        save_path='gtmocol_pretrained.pt'
    )
    
    print("Pre-training completed successfully!")

if __name__ == "__main__":
    main()
