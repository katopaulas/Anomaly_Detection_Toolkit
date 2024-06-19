from data_curation import get_data

def train(DF):

    #TRAIN/TEST split
    X,Y= DF.iloc[:,:-1],DF.iloc[:,-1]
    thresh = int(0.8*len(X))
    x_train,y_train,x_test,y_test = X[:thresh],Y[:thresh], X[thresh:],Y[thresh:]


    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import SGDOneClassSVM
    from sklearn.pipeline import make_pipeline
    outliers_fraction = 0.15

    clf = SGDOneClassSVM(nu=outliers_fraction,
                         shuffle=True,
                         fit_intercept=True,
                         random_state=42,
                         tol=1e-6,
     )



    def kernelize_data(x):
        feature_map_nystroem = Nystroem(gamma=.2,
                                        random_state=1,
                                        n_components=300)
        data_transformed = feature_map_nystroem.fit_transform(x)
        return data_transformed,feature_map_nystroem

    data_transformed, feature_map = kernelize_data(x_train)
    clf.fit(data_transformed)


    train_pred = clf.predict(data_transformed)
    train_pred[train_pred==-1]=0
    data_transformed = feature_map.transform(x_test)
    test_pred = clf.predict(data_transformed)
    test_pred[test_pred==-1] = 0

    #metrics
    from sklearn.metrics import roc_auc_score
    roc_auc_score(y_train,train_pred)
    roc_auc_score(y_test,test_pred)


trainable_df = get_data()
train(DF)








