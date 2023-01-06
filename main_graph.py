import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
import random
import pandas as pd
from utils_graph import *
from models_graph import GNNs
import argparse
import dgl
import os

parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=str, default='Regression',
                    choices=['Classification','Regression'])
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--src_vocab_size', type=int, default=15) # number of beads
parser.add_argument('--embed_dim', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--model', type=str, default='GraphSAGE',choices=['GCN','GAT','GraphSAGE'])
parser.add_argument('--GraphSAGE_aggregator', type=str, default='lstm',choices=['mean', 'gcn', 'pool', 'lstm'])
parser.add_argument('--GAT_heads', type=int, default=4)
parser.add_argument('--model_path', type=str, default='model_val_best.pt')

args = parser.parse_args()

args.model_path='{}_lr_{}_hid_{}_bs_{}.pt'.format(args.model,args.lr,args.hidden,args.batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'    
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)          

def main():
    
    if args.task_type == 'Classification':
        train_dataset, train_label = dgl.load_graphs('Graphical_Peptides/dgl_train_graphs_cla.bin')
        valid_dataset, valid_label = dgl.load_graphs('Graphical_Peptides/dgl_valid_graphs_cla.bin')
        test_dataset, test_label = dgl.load_graphs('Graphical_Peptides/dgl_test_graphs_cla.bin')
        train_label = train_label['labels'].long()
        valid_label = valid_label['labels'].long()
        test_label = test_label['labels'].long()
    elif args.task_type == 'Regression':
        train_dataset, train_label = dgl.load_graphs('Graphical_Peptides/dgl_train_graphs.bin')
        valid_dataset, valid_label = dgl.load_graphs('Graphical_Peptides/dgl_valid_graphs.bin')
        test_dataset, test_label = dgl.load_graphs('Graphical_Peptides/dgl_test_graphs.bin')
        train_label = train_label['labels'].float()
        valid_label = valid_label['labels'].float()
        test_label = test_label['labels'].float()

    dataset = MyDataSet(train_dataset,train_label)
    train_loader = Data.DataLoader(dataset, args.batch_size, True,
                         collate_fn=collate)
    valid_set = MyDataSet(valid_dataset,valid_label)
    val_graphs,val_labels = collate(valid_set)
    
    test_set = MyDataSet(test_dataset,test_label)
    test_graphs,test_labels = collate(test_set)

    valid_mse_saved = 100
    valid_acc_saved = 0
    loss_mse = torch.nn.MSELoss()
    # loss_cla = torch.nn.BCELoss()

    if args.task_type == 'Classification':
        model = GNNs(args, 2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    elif args.task_type == 'Regression':
        model = GNNs(args, 1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for graphs,labels in train_loader:
            graphs = graphs.to(device)
            labels = labels.to(device)
            outputs = model(graphs)
            
            if args.task_type == 'Classification':
                loss = F.nll_loss(outputs, labels)
                # loss = loss_cla(outputs, labels)
            elif args.task_type == 'Regression':
                loss = loss_mse(outputs, labels)

            # print('Epoch:',epoch,', Training Loss:',loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 1 == 0:
            model.eval()
            predict = model(val_graphs)
            
            if args.task_type == 'Classification':
                valid_acc = accuracy(predict, val_labels)
                if valid_acc_saved < valid_acc:
                    valid_acc_saved = valid_acc
                    print('Epoch:',epoch+1)
                    print('Valid Performance:',valid_acc.item())
                    torch.save(model.state_dict(),args.model_path)
                    print('Test Performance:',accuracy(model(test_graphs),test_labels).item())

            elif args.task_type == 'Regression':
                valid_mse = loss_mse(predict,val_labels).item()
                if valid_mse_saved > valid_mse:
                    valid_mse_saved = valid_mse
                    print('Epoch:',epoch+1)
                    print('Valid Performance:',valid_mse)
                    torch.save(model.state_dict(),args.model_path)
                    print('Test Performance:',loss_mse(model(test_graphs),test_labels).item())

    if args.task_type == 'Classification':
        predict = []
        model_load = GNNs(args, 2).to(device)
        checkpoint = torch.load(args.model_path)
        model_load.load_state_dict(checkpoint)
        model_load.eval()

        outputs = model_load(test_graphs)
        predict_test = outputs.max(1)[1].type_as(test_label)
        predict = predict + predict_test.cpu().detach().numpy().tolist()

        df_test_save = pd.DataFrame()
        labels = test_label.squeeze(1).tolist()
        df_test_seq = pd.read_csv('Sequential_Peptides/test_seqs_cla.csv')
        df_test_save['feature'] = df_test_seq['Feature']
        df_test_save['predict'] = predict
        df_test_save['label'] = labels
        test_acc = accuracy(model_load(test_graphs),test_labels).item()
        df_test_save['Acc'] = test_acc

        from sklearn.metrics import precision_score, recall_score, f1_score
        df_test_save['Precision'] = precision_score(test_label.cpu(),predict_test.cpu().detach())
        df_test_save['Recall'] = recall_score(test_label.cpu(),predict_test.cpu().detach())
        df_test_save['F1-score'] = f1_score(test_label.cpu(),predict_test.cpu().detach())

        df_test_save.to_csv('results_graph/Test_cla_{}_Acc_{}_lr_{}_hid_{}_bs_{}.csv'.format(args.model,test_acc,args.lr,args.hidden,args.batch_size))

    if args.task_type == 'Regression':
        predict = []
        model_load = GNNs(args, 1).to(device)
        checkpoint = torch.load(args.model_path)
        model_load.load_state_dict(checkpoint)
        model_load.eval()

        outputs = model_load(test_graphs)
        predict = predict + outputs.squeeze(1).cpu().detach().numpy().tolist()

        df_test_save = pd.DataFrame()
        labels = test_label.squeeze(1).tolist()
        df_test_seq = pd.read_csv('Sequential_Peptides/test_seqs.csv')
        df_test_save['feature'] = df_test_seq['Feature']
        df_test_save['predict'] = predict
        df_test_save['label'] = labels
        error = []
        for i in range(len(labels)):
            error.append(labels[i]-predict[i])
        absError = []
        squaredError = []
        for val in error:
            absError.append(abs(val))
            squaredError.append(val*val)
        MSE = sum(squaredError)/len(squaredError)
        MAE = sum(absError)/len(absError)

        from sklearn.metrics import r2_score
        R2 = r2_score(test_label.cpu(),outputs.cpu().detach())
        
        df_test_save['MSE'] = squaredError
        df_test_save['MAE'] = absError
        df_test_save['MSE_ave'] = MSE
        df_test_save['MAE_ave'] = MAE
        df_test_save['R2'] = R2
        df_test_save.to_csv('results_graph/Test_reg_{}_MAE_{}_lr_{}_hid_{}_bs_{}.csv'.format(args.model,MAE,args.lr,args.hidden,args.batch_size))

    os.remove(args.model_path)

if __name__ == '__main__':
    main()