import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pandas as pd
from utils_seq import *
from models_mlp import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=str, default='Regression',
                    choices=['Classification','Regression'])
parser.add_argument('--seed', type=int, default=5, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.2,
                    help='Initial learning rate.')
# parser.add_argument('--hidden', type=int, default=256,
                    # help='Number of hidden units.')
parser.add_argument('--src_vocab_size', type=int, default=21) # number of amino acids + 'Empty'
parser.add_argument('--src_len', type=int, default=10)
# parser.add_argument('--embed_dim', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--model', type=str, default='MLP')
parser.add_argument('--model_path', type=str, default='model_val_best.pt')

args = parser.parse_args()
args.d_model = 512  # Embedding size
args.model_path='{}_lr_{}_bs_{}.pt'.format(args.model,args.lr,args.batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'    
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)  

def main():

    if args.task_type == 'Classification':
        df_train = pd.read_csv('Sequential_Peptides/train_seqs_cla.csv')
        df_valid = pd.read_csv('Sequential_Peptides/valid_seqs_cla.csv')
        df_test = pd.read_csv('Sequential_Peptides/test_seqs_cla.csv')
        train_label = torch.Tensor(np.array(df_train["Label"])).long()
        valid_label = torch.Tensor(np.array(df_valid["Label"])).long().to(device)
        test_label = torch.Tensor(np.array(df_test["Label"])).long().to(device)
    elif args.task_type == 'Regression':
        df_train = pd.read_csv('Sequential_Peptides/train_seqs.csv')
        df_valid = pd.read_csv('Sequential_Peptides/valid_seqs.csv')
        df_test = pd.read_csv('Sequential_Peptides/test_seqs.csv')
        train_label = torch.Tensor(np.array(df_train["Label"])).unsqueeze(1).float()
        valid_label = torch.Tensor(np.array(df_valid["Label"])).unsqueeze(1).float().to(device)
        test_label = torch.Tensor(np.array(df_test["Label"])).unsqueeze(1).float().to(device)
    
    train_feat = np.array(df_train["Feature"])
    valid_feat = np.array(df_valid["Feature"])
    test_feat = np.array(df_test["Feature"])

    train_enc_inputs = make_data(train_feat,args.src_len)
    valid_enc_inputs = make_data(valid_feat,args.src_len).to(device)
    test_enc_inputs = make_data(test_feat,args.src_len).to(device)

    train_loader = Data.DataLoader(MyDataSet(train_enc_inputs,train_label), args.batch_size, True)

    valid_mse_saved = 100
    valid_acc_saved = 0
    loss_mse = torch.nn.MSELoss()

    model = MLP(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(args.epochs):
        model.train()
        for enc_inputs,labels in train_loader:
            enc_inputs = enc_inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(enc_inputs)

            if args.task_type == 'Classification':
                loss = F.nll_loss(outputs, labels)
            elif args.task_type == 'Regression':
                loss = loss_mse(outputs, labels)

            # print('Epoch:','%04d' % (epoch+1), 'loss =','{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 1 == 0:
            model.eval()
            predict = model(valid_enc_inputs)
            
            if args.task_type == 'Classification':
                valid_acc = accuracy(predict, valid_label)
                if valid_acc_saved < valid_acc:
                    valid_acc_saved = valid_acc
                    print('Epoch:',epoch+1)
                    print('Valid Performance:',valid_acc.item())
                    torch.save(model.state_dict(),args.model_path)

            elif args.task_type == 'Regression':
                valid_mse = loss_mse(predict,valid_label).item()
                if valid_mse_saved > valid_mse:
                    valid_mse_saved = valid_mse
                    print('Epoch:',epoch+1)
                    print('Valid Performance:',valid_mse)
                    torch.save(model.state_dict(),args.model_path)
    
    predict = []

    model_load = MLP(args).to(device)
    checkpoint = torch.load(args.model_path)
    model_load.load_state_dict(checkpoint)
    model_load.eval()
    
    outputs = model_load(test_enc_inputs)

    if args.task_type == 'Classification':
        
        predict_test = outputs.max(1)[1].type_as(test_label)
        predict = predict + predict_test.cpu().detach().numpy().tolist()

        df_test_save = pd.DataFrame()
        labels = test_label.tolist()
        df_test_seq = pd.read_csv('Sequential_Peptides/test_seqs_cla.csv')
        df_test_save['feature'] = df_test_seq['Feature']
        df_test_save['predict'] = predict
        df_test_save['label'] = labels
        test_acc = accuracy(model_load(test_enc_inputs),test_label).item()
        df_test_save['Acc'] = test_acc
        from sklearn.metrics import precision_score, recall_score, f1_score
        df_test_save['Precision'] = precision_score(test_label.cpu(),predict_test.cpu().detach())
        df_test_save['Recall'] = recall_score(test_label.cpu(),predict_test.cpu().detach())
        df_test_save['F1-score'] = f1_score(test_label.cpu(),predict_test.cpu().detach())

        df_test_save.to_csv('results_seq/Test_cla_{}_Acc_{}_lr_{}_bs_{}.csv'.format(args.model,test_acc,args.lr,args.batch_size))

    if args.task_type == 'Regression':

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
        df_test_save.to_csv('results_seq/Test_reg_{}_MAE_{}_lr_{}_bs_{}.csv'.format(args.model,MAE,args.lr,args.batch_size))

    os.remove(args.model_path)

if __name__ == '__main__':
    main()