
import data_curation
from ml import  train_svc, train_IF, train_LOF, train_nn, pca_fit, test_nn, test_clf

DATA_MANAGER = data_curation.data_manager()
CLASSIFIERS = ['IF','LOF','SVM','NN']#'pca',
            # NOVELTY                    OUTLIER
clf_dict = {'SVM': train_svc, 'IF':train_IF, 'LOF':train_LOF, 'NN':train_nn} #'pca': pca_fit,
clf_testers = {'SVM': test_clf,  'IF':test_clf, 'LOF':train_LOF, 'NN':test_nn}


#data ETL
#x_train,y_train,x_test,y_test = DATA_MANAGER.get_trainable_data()
x_train,y_train,x_test,y_test = DATA_MANAGER.get_trainable_data(norm=1)

for clf_name in CLASSIFIERS:
    
    trainer = clf_dict[clf_name]
    tester  = clf_testers[clf_name]
    
    #if clf_name in ['SVM','NN']:
        
    y_pred_proba,model = trainer(x_train,y_train)
    print('Predict\n')
    tester(x_test,y_test,model,name=clf_name)
    