import argparse
from re import M

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=2000, type=int, help="Epochs.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--val_batch_size", default=1000, type=int, help="Evaluation batch size.")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
parser.add_argument("--no_emb", default=False, type=bool, help="Whether to include embedding information")
parser.add_argument("--emb_name", default="t5", type=str, help="Embedding name : onehot, bert, xlnet")
parser.add_argument("--fold", default="", type=str, help="Cross-Validation fold number (ex : fold_0_")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of epochs")
parser.add_argument("--output_folder", default="", type=str, help="Output folder : Training sets | Testing sets")
parser.add_argument("--cutoff", default=6, type=int, help="Graph cutoff value : 4,6,8,10,12")
parser.add_argument("--ligand", default="", type=str, help="Ligand")
parser.add_argument("--model_archi", default="shallow", type=str, help="Model class : shallow,deep")
parser.add_argument("--model_class", default="gcn", type=str, help="Model class : gcn,gat")


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

assert torch.cuda.is_available() , "CUDA is NOT available !!"

DEVICE="cuda"

args = parser.parse_args([] if "__file__" not in globals() else None)
# 
import warnings
warnings.filterwarnings("ignore")


# 
from sklearn.metrics import recall_score,accuracy_score,precision_score
from sklearn.metrics import matthews_corrcoef as MCC


def get_metric(pred,labels,name):
    # Accuracy
    if name=="accuracy":
        return accuracy_score(np.array(labels.cpu()), np.array(pred.cpu()))
    
    # Recall
    if name=="recall":
        return recall_score(np.array(labels.cpu()), np.array(pred.cpu()),average='macro')

    # Precision
    if name=="precision":
        return precision_score(np.array(labels.cpu()), np.array(pred.cpu()),average='macro')
    # Matthews Correlation Coefficient (MCC)
    if name=="mcc":
        return MCC(np.array(labels.cpu()), np.array(pred.cpu()))




##__________________ Load data and make batches _____________________________________

# Data folder
data_folder=f"/home/files/yu_gnn"
EMB_NAME=args.emb_name
FEAT_SIZES={"onehot":21,
            "bert":1024,
            "xlnet":1024,
            "t5":1024,
            "aaindex":566}


# else:

TEST_MODEL=False


TRAIN_MODE="Training"
VAL_MODE="Val"
TEST_MODE="Validation"
# LIGAND=args.ligand
CUTOFF=args.cutoff

MODES=[TRAIN_MODE,TEST_MODE]

# _____________________________ Model Architecture ______________________________
from dgllife.model.gnn.gat import GAT
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.gnn.gcn import GCN

def softmax_fct(h):
    return F.softmax(h,1)


n_feats=FEAT_SIZES[EMB_NAME]

class GNN(nn.Module):
    
    def __init__(self, layers):
        super(GNN, self).__init__()
        n_layers=len(layers)
        ACTIVATIONS=[F.relu for _ in range(n_layers)]
        DROPOUTS=[args.dropout for _ in range(n_layers)]
        NUM_HEADS=None
        if args.model_class=="gat":
            self.gnn=GAT(n_feats, hidden_feats=layers, activations=ACTIVATIONS, 
                        num_heads=NUM_HEADS, feat_drops =DROPOUTS)
        if args.model_class=="gcn":
            self.gnn=GCN(n_feats,hidden_feats=layers,
                     residual=[True]*n_layers, batchnorm=[True]*n_layers,
                     dropout=[args.dropout]*n_layers)
    
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


def get_graphs(mode,LIGAND):
    if mode=="Training":
        folder="Training_sets"
    if mode=="Validation":
        folder="Testing_sets"
    with zipfile.ZipFile(data_folder+f'/{folder}/all_embs_{LIGAND}_{mode}_th_{CUTOFF}.zip','r') as thezip:
        graphs=[]
        for file in thezip.namelist():
            dest=f"{data_folder}/{args.model_class}_{EMB_NAME}/{file}"
            thezip.extract(file,f"{data_folder}/{args.model_class}_{EMB_NAME}")
            # graphs+=[dgl.load_graphs(dest)[0][0]]
            g=pickle.load(open(dest,'rb'))
            # g.ndata["feat"]=torch.zeros_like(g.ndata["feat"])
            graphs+=[g]
            os.remove(dest)
        return graphs


print(f"Cutoff value : {CUTOFF}")

PREFIX=""
if EMB_NAME=="bert":
    PREFIX='bert_'
if EMB_NAME=="aaindex":
    PREFIX='pc_'


def get_batch(GRAPHS,a,b,mode):
    batch_graph=[]
    for graph in GRAPHS[mode][a:b]:
        batch_graph+=[graph]
    batch_graph=dgl.batch(batch_graph)
    batch_graph=batch_graph.to(DEVICE)
    features = batch_graph.ndata[PREFIX+'feat'].to(DEVICE)
    labels = batch_graph.ndata['label'].to(DEVICE)
    w0=0
    w1=0
    if mode==TRAIN_MODE:
        n0=int((labels==0).sum())
        n1=int((labels==1).sum())
        w0=n1/(n1+n0)
        w1=n0/(n1+n0)

    return batch_graph,features,labels,w0,w1

# Function to Train the model    

def train(GRAPHS,model,metric_name):
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0003,weight_decay=1e-5)
    best_score=0
    # best_score=1
    RESULTS={"Epoch":[],"Train MCC":[],"Val MCC":[]}
    for e in range(args.epochs+1):
        evaluate=(e%args.evaluate_each==0)

        if evaluate:
            train_pred=torch.Tensor([]).to(DEVICE)
            train_labels=torch.Tensor([]).to(DEVICE)

        model.train()
        a=0
        while a <len(GRAPHS[TRAIN_MODE]):
            b=a+args.batch_size
            # load batch
            g_train,features,labels,w0,w1=get_batch(GRAPHS,a,b,TRAIN_MODE)
            # Forward propagation
            logits = model(g_train, features)
            if evaluate:
                # Get predictions and labels for evaluation
                train_pred = torch.concat([train_pred,logits.argmax(1)]) 
                train_labels = torch.concat([train_labels,labels]) 
            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            weight=(torch.Tensor([w0,w1])).to(DEVICE)
            loss=nn.CrossEntropyLoss(weight)(logits.float(), labels.reshape(-1,).long())
            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            a=b

        t = 1000 * time.time() # current time in milliseconds
        random.seed(int(t) % 2**32)
        random.shuffle(GRAPHS[TRAIN_MODE])
        
        if evaluate:
            model.eval()
        # Get evaluation metric       
            train_metric=get_metric(train_pred,train_labels,metric_name)
            RESULTS["Epoch"].append(e+1)
            RESULTS["Train MCC"].append(train_metric)
            val_pred=torch.tensor([]).to(DEVICE)
            val_labels=torch.tensor([]).to(DEVICE)
            a=0
            with torch.no_grad():
                while a< len(GRAPHS[VAL_MODE]):
                    b=a+args.val_batch_size
                    # load batch
                    g_val,features,labels,_,_=get_batch(GRAPHS,a,b,VAL_MODE)
                    # Forward propagation
                    logits = model(g_val, features)
                    # val_loss+=nn.CrossEntropyLoss()(logits.float(), labels.reshape(-1,).long())
                    # Get predictions and labels for evaluation
                    val_pred = torch.concat([val_pred,logits.argmax(1)]) 
                    val_labels = torch.concat([val_labels,labels])
                    a=b

            val_metric=get_metric(val_pred,val_labels,metric_name)
            if val_metric>best_score:
                print("Saving model...")
                best_score=val_metric
                torch.save(model.state_dict(),f"/home/files/{EMB_NAME}_{LIGAND}_th_{CUTOFF}_{args.model_class}_{args.model_archi}_fold{args.fold}_model.pt")
                # torch.save(model.state_dict(),f"/home/files/{EMB_NAME}_{LIGAND}_no_graph.pt")
            RESULTS["Val MCC"].append(val_metric)
            print('In epoch {}, loss: {:.3f}, train {} : {:.3f} , val {} : {:.3f}'.format(
            e, loss,metric_name,train_metric,metric_name ,val_metric))

    return RESULTS
                
            
        


# Configure training



if args.no_emb:
    print("No Embedding !!")
else:
    print("Embedding : ",EMB_NAME)

print(f"CV Fold : {args.fold}")

# if args.no_graph:
#     print("No Graph information !!")
if args.model_archi=="shallow":
    LAYERS=[512]
if args.model_archi=="deep":
    LAYERS=[512]*6
print(f"Model class :  {args.model_class.upper()}")

print("Layers : ",[n_feats]+LAYERS+[2])
print("Dropout rate : ",args.dropout)
# if args.batch_norm:
#     print("Using Batch Normalization")
# if args.residual:
#     print("Using Residual connections")
print(f"Training for {args.epochs} Epochs ")
print("Batch size : ",args.batch_size)




if args.ligand=="":
    LIGANDS=["ADP", "AMP", "ATP", "CA", "DNA", "FE", "GDP", "GTP", "HEME", "MG", "MN", "ZN"]
else:
    LIGANDS=[args.ligand]

# with open(f"/home/files/test_results_attention_cutoff_{CUTOFF}.txt","w") as test_file:
with open(f"/home/files/test_results_seq_emb.txt","w") as test_file:
    for LIGAND in LIGANDS:
        print(f"Ligand type : {LIGAND}")

        # Loading data
        # try: 
        print("Loading graphs...")
        GRAPHS={mode:get_graphs(mode,LIGAND) for mode in MODES}
        print(LIGAND,len(GRAPHS["Training"]),len(GRAPHS["Validation"]))
        # random.seed(89)
        if args.fold!="":
            X=list(range(len(GRAPHS["Training"])))
            folds=KFold(n_splits=5,random_state=10,shuffle=True)
            for fold_idx, (train_idx, test_idx) in enumerate(folds.split(X)):
                if fold_idx==int(args.fold):
                    GRAPHS["Val"]=[GRAPHS["Training"][i] for i in test_idx]

        else:
            random.seed(10)
            GRAPHS["Val"]=random.sample(GRAPHS["Training"],len(GRAPHS["Validation"]))

        GRAPHS["Training"]=[x for x in GRAPHS["Training"] if x not in GRAPHS["Val"]]


        for mode in GRAPHS.keys():
            print(f"{mode} : {len(GRAPHS[mode])} graphs")

        model = GNN(LAYERS).cuda()
        # Train the model
        print("Training model")
        
        RESULTS=train(GRAPHS,model,metric_name="mcc")
        # Get results
        # Save training history
        # pd.DataFrame(RESULTS).to_csv(f"/home/files/results_{EMB_NAME}_{LIGAND}_th_{CUTOFF}.csv")

        # Final Test of the model
        if TEST_MODEL:
            model = GNN(args.layers).cuda()
            model.load_state_dict(torch.load(f"/home/files/{EMB_NAME}_{LIGAND}_th_{CUTOFF}_{args.model_class}_{args.model_archi}_fold{args.fold}_model.pt"))
            # model.load_state_dict(torch.load(f"/home/files/{EMB_NAME}_{LIGAND}_no_graph.pt"))
            model.eval()
            # # Test on test set
            print("Evaluating on test set ...")
            test_pred=torch.tensor([]).to(DEVICE)
            test_labels=torch.tensor([]).to(DEVICE)
            a=0
            while a <len(GRAPHS[TEST_MODE]):
                b=a+args.val_batch_size
                # load batch
                g_test,features,labels,_,_=get_batch(GRAPHS,a,b,TEST_MODE)
                # Forward propagation
                with torch.no_grad():
                    logits = model(g_test, features)
                # Get predictions and labels for evaluation
                test_pred = torch.concat([test_pred,logits.argmax(1)]) 
                test_labels = torch.concat([test_labels,labels])
                a=b
            print(f"Ligand Type : {LIGAND}  ----->  Test MCC : {get_metric(test_pred,test_labels,'mcc')}")
            # test_file.write(f"Ligand Type : {LIGAND}  ----->  Test MCC : {get_metric(test_pred,test_labels,'mcc')}")
            # test_file.write("\n")





