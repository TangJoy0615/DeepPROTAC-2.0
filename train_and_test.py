import logging
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, auc, confusion_matrix, accuracy_score, precision_recall_curve, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from collections import defaultdict
import math

def valids(model, test_loader, device):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        model.eval()
        y_true = []
        y_pred = []
        y_score = []
        loss = []
        iteration = 0
        for data_sample in test_loader:
            y = data_sample['label'].to(device)
            outputs = model(
                data_sample['ligase_ligand'].to(device),
                data_sample['ligase_pocket'].to(device),
                data_sample['target_ligand'].to(device),
                data_sample['target_pocket'].to(device),
                data_sample['smiles'].to(device),
                data_sample['smiles_length'],
            )
            loss_val = criterion(outputs, y)
            loss.append(loss_val.item())
            y_score = y_score + torch.nn.functional.softmax(outputs,1)[:,1].cpu().tolist()
            y_pred = y_pred + torch.max(outputs,1)[1].cpu().tolist()
            y_true = y_true + y.cpu().tolist()
            iteration += 1
        model.train()
    return sum(loss)/iteration, accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_score)




def train(model, lr=0.0001, epoch=30, train_loader=None, valid_loader=None, test_loader=None, device=None, writer=None, LOSS_NAME=None, batch_size = None):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    _ = valids(model, valid_loader, device)
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    best_epo = 0
    print("Start Training")
    for epo in range(epoch):
        total_num = 0
        for data_sample in train_loader:
            outputs = model(
                data_sample['ligase_ligand'].to(device),
                data_sample['ligase_pocket'].to(device),
                data_sample['target_ligand'].to(device),
                data_sample['target_pocket'].to(device),
                data_sample['smiles'].to(device),
                data_sample['smiles_length'],
            )
            total_num += batch_size
            y = data_sample['label'].to(device)
            loss = criterion(outputs, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()

        logging.info('Train epoch %d, loss: %.4f' % (epo, running_loss/total_num))
        val_loss, val_acc, val_auroc = valids(model, valid_loader, device)
        logging.info(f'Valid epoch {epo} loss:{val_loss}, acc: {val_acc}, auroc: {val_auroc}')
        test_loss, test_acc, test_auroc = valids(model, test_loader, device)
        logging.info(f'Test epoch {epo} loss:{test_loss}, acc: {test_acc}, auroc: {test_auroc}')
        running_loss = 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epo = epo
            # torch.save(model, f"model/{LOSS_NAME}.pt")

    logging.info(f'The best model is in epoch {best_epo}')
    logging.info(f'The best acc is in epoch {best_val_acc}')

    torch.save(model, f"model/{LOSS_NAME}.pt")
    print("Finish Training")
    return model

