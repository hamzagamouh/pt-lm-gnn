# sbatch --job-name bert_4 --output bert_4.txt predict_bs_attention_yu.sh --cutoff 4 --fold 0 --emb_name bert
# sbatch --job-name bert_6 --output bert_6.txt predict_bs_attention_yu.sh --cutoff 6 --fold 0 --emb_name bert
# sbatch --job-name bert_10 --output bert_10.txt predict_bs_attention_yu.sh --cutoff 10 --fold 0 --emb_name bert


print("HELLO !")
import os
import time
import numpy as np
import pandas as pd
import pickle
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import zipfile
import random
import scipy
from sklearn.model_selection import KFold
 
import warnings
warnings.filterwarnings("ignore")


# 
from sklearn.metrics import recall_score,accuracy_score,precision_score,roc_auc_score
from sklearn.metrics import matthews_corrcoef as MCC


def get_metric(labels,pred,name):
    # Accuracy
    if name=="roc-auc":
        return roc_auc_score(labels,pred)
    # Recall
    if name=="recall":
        return recall_score(labels,pred)
    # Precision
    if name=="precision":
        return precision_score(labels,pred)
    # Matthews Correlation Coefficient (MCC)
    if name=="mcc":
        return MCC(labels, pred)






# _____________________________ Model Architecture ______________________________
from dgllife.model.gnn.gat import GAT
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.gnn.gcn import GCN

def softmax_fct(h):
    return F.softmax(h,1)


class GNN(nn.Module):
    
    def __init__(self, model_class,layers,n_feats=1024):
        super(GNN, self).__init__()
        n_layers=len(layers)
        ACTIVATIONS=[F.relu for _ in range(n_layers)]
        DROPOUTS=[0.5 for _ in range(n_layers)]
        NUM_HEADS=None
        if model_class=="gat":
            self.gnn=GAT(n_feats, hidden_feats=layers, activations=ACTIVATIONS, 
                        num_heads=NUM_HEADS, feat_drops =DROPOUTS)
        if model_class=="gcn":
            self.gnn=GCN(n_feats,hidden_feats=layers,
                        residual=[True]*n_layers, batchnorm=[True]*n_layers,
                        dropout=DROPOUTS)
    
        self.dense1=nn.Linear(layers[-1], 2)
        # self.dense1=nn.Linear(n_feats, 64)
        # self.dense2=nn.Linear(64, 64)
        # self.dense3=nn.Linear(64, 64)
        # self.dense4=nn.Linear(64, 2)

    def forward(self, g, in_feat):
        # edge_feats=torch.ones(g.num_edges(),1).to(DEVICE)
        h=self.gnn(g, in_feat)
        # h=torch.concat([h,in_feat],axis=1)
        # h=in_feat
        h=self.dense1(h)
        # h = F.relu(h)
        # h=F.dropout(h,args.dropout)
        # h=self.dense2(h)
        # h = F.relu(h)
        # h=F.dropout(h,args.dropout)
        # h=self.dense3(h)
        # h = F.relu(h)
        # h=F.dropout(h,args.dropout)
        # h=self.dense4(h)
        h = F.softmax(h,1)
        return h
# ____________________________ Util functions _____________________________

def majority_vote(pred):
    ensemble_pred=[]
    for x in pred:
        n0=(x==0).sum()
        n1=(x==1).sum()
        ensemble_pred+=[int(n0<n1)]
    
    return np.array(ensemble_pred).reshape(-1,1)


def to_numpy(t):
    return t.detach().cpu().numpy().reshape(-1,1)


##__________________ Load data and make batches _____________________________________

if __name__=='__main__':
    import argparse
    from re import M

    assert torch.cuda.is_available() , "CUDA is NOT available !!"

    DEVICE="cuda"

    args = parser.parse_args([] if "__file__" not in globals() else None)

    parser = argparse.ArgumentParser()
    parser.add_argument("--val_batch_size", default=1000, type=int, help="Evaluation batch size.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
    parser.add_argument("--no_emb", default=False, type=bool, help="Whether to include embedding information")
    parser.add_argument("--emb_name", default="aaindex", type=str, help="Embedding name : t5, bert, aaindex")
    parser.add_argument("--fold", default=0, type=str, help="Cross-Validation fold number (ex : fold_0_")
    parser.add_argument("--output_folder", default="", type=str, help="Output folder : Training sets | Testing sets")
    parser.add_argument("--ligand", default="", type=str, help="Ligand")

    # Data folder
    data_folder=f"/home/files/yu_gnn"
    EMB_NAME=args.emb_name
    FEAT_SIZES={"onehot":21,
                "bert":1024,
                "xlnet":1024,
                "t5":1024,
                "aaindex":566}

    n_feats=FEAT_SIZES[EMB_NAME]



    def get_graphs(LIGAND,CUTOFF):
        mode="Validation"
        folder="Testing_sets"
        with zipfile.ZipFile(data_folder+f'/{folder}/all_embs_{LIGAND}_{mode}_th_{CUTOFF}.zip','r') as thezip:
            graphs=[]
            for file in thezip.namelist():
                dest=f"{data_folder}/fold{args.fold}/{file}"
                thezip.extract(file,f"{data_folder}/fold{args.fold}")
                # graphs+=[dgl.load_graphs(dest)[0][0]]
                g=pickle.load(open(dest,'rb'))
                # g.ndata["feat"]=torch.zeros_like(g.ndata["feat"])
                graphs+=[g]
                os.remove(dest)
            return graphs


    PREFIX=""
    if EMB_NAME=="bert":
        PREFIX='bert_'
    if EMB_NAME=="aaindex":
        PREFIX='pc_'


    def get_batch(GRAPHS,a,b):
        batch_graph=[]
        for graph in GRAPHS[a:b]:
            batch_graph+=[graph]

        batch_graph=dgl.batch(batch_graph)
        batch_graph=batch_graph.to(DEVICE)
        features = batch_graph.ndata[PREFIX+'feat'].to(DEVICE)
        labels = batch_graph.ndata['label'].to(DEVICE)

        return batch_graph,features,labels




    if args.no_emb:
        print("No Embedding !!")
    else:
        print("Embedding : ",EMB_NAME)

    print("Dropout rate : ",args.dropout)





    if args.ligand=="":
        LIGANDS=["ADP", "AMP", "ATP", "CA", "DNA", "FE", "GDP", "GTP", "HEME", "MG", "MN", "ZN"]
    else:
        LIGANDS=[args.ligand]

    CUTOFFS=[4,6,8,10]

    METRICS=["roc-auc","precision","recall","mcc"]

    DF={"Ligand":[],"Embedding":[],"Model":[],"Cutoff":[]}

    for metric in METRICS:
        DF[metric]=[]

    TH=0.5

    for LIGAND in LIGANDS:
        for MODEL_CLASS in ["gat"]:
            MODEL_ARCHI = "shallow"
            if MODEL_ARCHI=="shallow":
                LAYERS=[512]
            elif MODEL_ARCHI=="deep":
                LAYERS=[512]*6
            else:
                LAYERS=[]
            print("Model class",MODEL_CLASS)
            print("Model archi",MODEL_ARCHI)
            all_preds=[]
            all_probs=[]
            # for FOLD in range(5):
            for CUTOFF in CUTOFFS:
                FOLD=0
                if CUTOFF==8:
                    FOLD=""
                print(f"Ligand type : {LIGAND}")
                print("Cutoff : ",CUTOFF)

                # Loading data
                # try: 
                print("Loading test graphs...")
                GRAPHS=get_graphs(LIGAND,CUTOFF)

                print(f"{len(GRAPHS)} graphs")
                model = GNN(MODEL_CLASS,LAYERS,n_feats).cuda()
                model.load_state_dict(torch.load(f"/home/files/{EMB_NAME}_{LIGAND}_th_{CUTOFF}_{MODEL_CLASS}_{MODEL_ARCHI}_fold{FOLD}_model.pt"))
                model.eval()
                # # Test on test set
                print("Evaluating on test set ...")
                test_pred=torch.tensor([]).to(DEVICE)
                test_labels=torch.tensor([]).to(DEVICE)
                a=0
                while a <len(GRAPHS):
                    b=a+args.val_batch_size
                    # load batch
                    g_test,features,labels=get_batch(GRAPHS,a,b)
                    # Forward propagation
                    with torch.no_grad():
                        logits = model(g_test, features)
                    # Get predictions and labels for evaluation
                    test_pred = torch.concat([test_pred,logits[:,1]]) 
                    test_labels = torch.concat([test_labels,labels])
                    a=b
                
                test_preds=to_numpy(test_pred>TH)
                test_probs=to_numpy(test_pred)
                test_labels=to_numpy(test_labels)

                DF["Ligand"]+=[LIGAND]
                DF["Embedding"]+=[EMB_NAME]
                DF["Model"]+=[MODEL_CLASS]
                DF["Cutoff"]+=[CUTOFF]
                # DF["Fold"]+=[FOLD]
                for metric in METRICS:
                    if metric!="roc-auc":
                        score=get_metric(test_labels,test_preds,metric)
                    else:
                        score=get_metric(test_labels,test_probs,metric)
                    print(metric,":",score)
                    DF[metric]+=[score]
                    
                all_preds+=[test_preds]
                all_probs+=[test_probs]

            DF["Ligand"]+=[LIGAND]
            DF["Embedding"]+=[EMB_NAME]
            DF["Model"]+=[MODEL_CLASS]
            DF["Cutoff"]+=["Ensemble"]
            all_preds=np.concatenate(all_preds,axis=1)
            all_probs=np.concatenate(all_probs,axis=1)
            all_preds=majority_vote(all_preds)
            all_probs=np.mean(all_probs,axis=1).reshape(-1,1)
            for metric in METRICS:
                if metric!="roc-auc":
                    score=get_metric(test_labels,all_preds,metric)
                else:
                    score=get_metric(test_labels,all_probs,metric)
                print(metric,":",score)
                DF[metric]+=[score]

            
    # pd.DataFrame(DF).to_csv(f"/home/files/yu_gnn_results/{LIGAND}_cutoffs_{EMB_NAME}.csv")
    # pd.DataFrame(DF).to_csv(f"/home/files/yu_gnn_results/{LIGAND}_no_graph_{EMB_NAME}.csv")
    pd.DataFrame(DF).to_csv(f"/home/files/yu_gnn_results/cutoffs_{EMB_NAME}.csv")




