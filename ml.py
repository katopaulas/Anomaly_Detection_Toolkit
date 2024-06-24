import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn.metrics import classification_report,balanced_accuracy_score,roc_auc_score
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import torch

def confusion_matrix_stats(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    confusion_m = confusion_matrix(y_true, y_pred)
    # FP = confusion_m.sum(axis=0) - np.diag(confusion_m)  
    # FN = confusion_m.sum(axis=1) - np.diag(confusion_m)
    # TP = np.diag(confusion_m)
    # TN = confusion_m.sum() - (FP + FN + TP)
    # print(TP,TN,FP,FN)
    #hardcomputed = (TP,TN,FP,FN)

    # or only for binary classification 
    TN = confusion_m[0][0]
    FN = confusion_m[1][0]
    TP = confusion_m[1][1]
    FP = confusion_m[0][1]
    return TP,TN,FP,FN#,hardcomputed

def metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred,output_dict =True)
    r_auc = roc_auc_score(y_true,  y_pred)

    df = pd.DataFrame(report).transpose()
    df['r_auc'] = r_auc
    
    df['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
    # #from sklearn.metrics import precision_recall_curve
    # precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    max_ones = len(y_true[y_true==1])
    TP,TN,FP,FN = confusion_matrix_stats(y_true, y_pred)
    print('Detected correctly TP ',TP, 'Max on data: ',max_ones )
    print('Detected extra FP',FP, '. Fraction of data they represent: ',FP/len(y_true))

    #df_1line_metrics = df_metrics_train[['precision','recall','r_auc', 'balanced_acc','f1-score']].iloc[1,:]
        
    return df

def mini_metric(y_true, y_pred):
    
    TP,TN,FP,FN = confusion_matrix_stats(y_true, y_pred)
    pct_anomaly = TN/(FP+TN) if FP>0 else 1
    
    print('F1: ',f1_score(y_true, y_pred, average='binary'))
    print('pct_anomalies detected:',pct_anomaly)
    print('FN ratio to TP:',FN/(FN+TP))
    print(('TP','TN','FP','FN'),TP,TN,FP,FN)
    print('ROCAUC:',roc_auc_score(y_true,  y_pred))
    print('-'*11)


def pca_fit(x, y, components = 45):
    pca = PCA(n_components=components)
    pca.fit(x)
    ex_var = pca.explained_variance_ratio_.cumsum()
    best = np.where(ex_var>0.98)[0]
    
    if len(best):
        components = best[0]
        pca = PCA(n_components=components)
        pca.fit(x)
    print('Performed pca with n=', components, 'components')
    return pca, components
    
   
def train_svc(x,y,name=''):
    outliers_fraction=0.15
    clf = SGDOneClassSVM(nu=outliers_fraction,
     )

    y_pred = clf.fit_predict(x)
    y_pred[y_pred==-1] = 0
    print(f'{name} finished')
    mini_metric(y,y_pred)
    return y_pred,clf

def train_LOF(x,y,clf=0,name='LOF'):
    clf = LocalOutlierFactor(n_neighbors=20)
    y_pred = clf.fit_predict(x)
    y_pred[y_pred==-1] = 0

    
    print(f'{name} finished')
    mini_metric(y,y_pred)
    return y_pred,clf

def train_IF(x,y):
    clf = IsolationForest(random_state=0).fit(x)
    y_pred = clf.predict(x)
    y_pred[y_pred==-1] = 0
    print('IF finished F1:')
    mini_metric(y,y_pred)
    return y_pred,clf

def train_nn(x,y,epochs=200):
    from nn import simple_nn,FocalLoss
    from tqdm import trange
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight('balanced',classes= np.unique(y),y= y)
    class_weights = torch.tensor(class_weights)
    
    model = simple_nn(x.shape[1],2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    criterion = FocalLoss(alpha=class_weights,gamma=2)
    
    x = torch.tensor(x.to_numpy()).float()
    y = torch.tensor(y.to_numpy()).float()
    model.train()
    for episode_number in (t:=trange(epochs)):
        y_pred_proba = model(x).squeeze()

        optimizer.zero_grad()
        loss = criterion(y_pred_proba, y)
        loss.backward()
        optimizer.step()
        t.set_description(f"epoch: {episode_number}, loss: {loss:.2f}")

    y_pred = model(x).argmax(dim=-1)
    #y_pred[y_pred > 0.8] = 1
    #y_pred[y_pred <= 0.8] = 0
    
    mini_metric(y,y_pred.detach().numpy())
    return y_pred,model

def test_clf(x_test,y_test,clf,name=''):
    y_pred = clf.predict(x_test)
    y_pred[y_pred==-1] = 0
    print(f'{name} finished:')
    mini_metric(y_test,y_pred)

def test_nn(x_test,y_test,model,name='NN'):
    x_test = torch.tensor(x_test.to_numpy()).float()
    y_pred = model(x_test).squeeze().argmax(dim=-1)
    print(f'{name} finished.')
    mini_metric(y_test,y_pred.detach().numpy())
        








