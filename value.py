import sys
import numpy as np
import torch
import os
import pickle
import logging
from torch.utils.data import DataLoader
from protacloader import PROTACSet, collater
from train_and_test import train, valids
from prepare_data import GraphData

BATCH_SIZE = 1
root = "test"
TRAIN_NAME = "test"
logging.basicConfig(filename="log/"+TRAIN_NAME+".log", filemode="w", level=logging.DEBUG)

def main():
    ligase_ligand = GraphData("ligase_ligand",root)
    ligase_pocket = GraphData("ligase_pocket",root)
    target_ligand = GraphData("target_ligand",root)
    target_pocket = GraphData("target_pocket",root)
    with open(os.path.join(target_pocket.processed_dir, "smiles.pkl"),"rb") as f:
        smiles = pickle.load(f)
    with open('data_list.pkl','rb') as f:
        name_list = pickle.load(f)
    label = torch.load(os.path.join(target_pocket.processed_dir, "label1.pt"))

    protac_set = PROTACSet(
        name_list,
        ligase_ligand, 
        ligase_pocket, 
        target_ligand, 
        target_pocket, 
        smiles, 
        label,
    )

    logging.info(f'Test data:{len(name_list)}')
    testloader = DataLoader(protac_set, batch_size=BATCH_SIZE, collate_fn=collater,drop_last=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('GATModel.pt')
    val_loss, val_acc, val_auroc= valids(model, testloader, device)
    logging.info(f'Valid loss:{val_loss}, acc: {val_acc}, auroc: {val_auroc}')


if __name__ == "__main__":
    main()