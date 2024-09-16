import sys
import numpy as np
import torch
import os
import pickle
import logging
from pathlib import  Path
from torch.utils.data import DataLoader
from protacloader import PROTACSet, collater


from model_GATdis import GraphConv, SmilesNet, ProtacModel
from train_and_test import train
from prepare_data import GraphData


# total
BATCH_SIZE = 1
EPOCH = 50
TRAIN_RATE = 0.8
VALID_RATE = 0.1
LEARNING_RATE = 0.0001
TRAIN_NAME = "test1"
w = True
depth = 2
root = "test1"
testmode = False
logging.basicConfig(filename="log/"+TRAIN_NAME+".log", filemode="w", level=logging.DEBUG)


def main():
    ligase_ligand = GraphData("ligase_ligand", root)
    ligase_pocket = GraphData("ligase_pocket", root)
    target_ligand = GraphData("target_ligand", root)
    target_pocket = GraphData("target_pocket", root)

    with open(os.path.join(target_pocket.processed_dir, "smiles.pkl"),"rb") as f:
        smiles = pickle.load(f)
    with open('data.pkl','rb') as f:
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
    data_size = len(protac_set)
    train_size = int(data_size * TRAIN_RATE)
    valid_size = int(data_size * VALID_RATE)
    test_size = data_size - train_size - valid_size

    if not testmode:
        valid_size = data_size // 10
        test_size = data_size // 10
        train_size = data_size - valid_size - test_size
    else:
        valid_size = 0
        test_size = data_size // 10
        train_size = data_size - valid_size - test_size
        

    logging.info(f"all data: {data_size}")
    logging.info(f"train data: {train_size}")
    logging.info(f"valid data: {valid_size}")
    logging.info(f"test data: {test_size}")
    train_dataset = torch.utils.data.Subset(protac_set, range(train_size))

    if not testmode:
        valid_dataset = torch.utils.data.Subset(protac_set, range(train_size, train_size+valid_size))
        test_dataset = torch.utils.data.Subset(protac_set, range(train_size+valid_size, data_size))

    else:
        valid_dataset = torch.utils.data.Subset(protac_set, range(train_size, data_size))
        test_dataset = torch.utils.data.Subset(protac_set, range(train_size, data_size))

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collater,drop_last=False, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collater,drop_last=False)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collater,drop_last=False)

    ligase_ligand_model = GraphConv(num_embeddings=10, depth=depth, w=w)
    ligase_pocket_model = GraphConv(num_embeddings=5, depth=depth, w=w)
    target_ligand_model = GraphConv(num_embeddings=10, depth=depth, w=w)
    target_pocket_model = GraphConv(num_embeddings=5, depth=depth, w=w)
    smiles_model = SmilesNet(batch_size=BATCH_SIZE)
    model = ProtacModel(
        ligase_ligand_model, 
        ligase_pocket_model,
        target_ligand_model,
        target_pocket_model,
        smiles_model,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train(
        model, 
        train_loader=trainloader, 
        valid_loader=validloader,
        test_loader=testloader,
        device=device,
        LOSS_NAME=TRAIN_NAME,
        batch_size=BATCH_SIZE,
        epoch=EPOCH,
        lr=LEARNING_RATE
    )

if __name__ == "__main__":
    Path('log').mkdir(exist_ok=True)
    Path('model').mkdir(exist_ok=True)
    main()
