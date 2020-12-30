import sklearn
import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from loaders import get_target_indexes

def kneighbors_classifier(model,dset_train, dset_test, classes=[0,1,2,3,4,5,6,7,8,9]):
    neigh = KNeighborsClassifier(n_neighbors=3)

    idx_train_subset = get_target_indexes(dset_train, classes, 10)
    X_train = dset_train.data[idx_train_subset].type(torch.float)
    X_train = torch.unsqueeze(X_train, 1)
    y_train = dset_train.targets[idx_train_subset].numpy()

    model.eval()
    with torch.no_grad():
        mu, logvar=model.encode(X_train.to(model.device))
        z=model.reparameterize(mu, logvar)
    Z_train = z.cpu().numpy()

    clf = KNeighborsClassifier(n_neighbors=3).fit(Z_train, y_train)

    y_pred = clf.predict(Z_train)
    train_acc = sklearn.metrics.accuracy_score(y_train, y_pred)
    print("Mean acc (train): ", train_acc)

    idx_test_subset = get_target_indexes(dset_test, classes, 10)
    X_test = dset_test.data[idx_test_subset].type(torch.float)
    X_test = torch.unsqueeze(X_test, 1)
    y_test = dset_test.targets[idx_test_subset].numpy()

    model.eval()
    with torch.no_grad():
        mu, logvar=model.encode(X_test.to(model.device))
        z=model.reparameterize( mu, logvar)
        Z_test = z.cpu().numpy()

    y_pred = clf.predict(Z_test)
    test_acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    print("Mean acc (test): ", test_acc)

    cm = sklearn.metrics.confusion_matrix(y_true = y_test, y_pred = y_pred)
    df_cm = pd.DataFrame(cm, index = [str(i) for i in classes],
                           columns = [str(i) for i in classes])

    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plt.show()

def linearregression_classifier(model,dset_train, dset_test, classes=[0,1,2,3,4,5,6,7,8,9]):

    idx_train_subset = get_target_indexes(dset_train, classes, 10)
    X_train = dset_train.data[idx_train_subset].type(torch.float)
    X_train = torch.unsqueeze(X_train, 1)
    y_train = dset_train.targets[idx_train_subset].numpy()

    model.eval()
    with torch.no_grad():
        mu, logvar=model.encode(X_train.to(model.device))
        z=model.reparameterize(mu, logvar)
    Z_train = z.cpu().numpy()
    
    clf = LogisticRegression(random_state=0).fit(Z_train, y_train)

    y_pred = clf.predict(Z_train)
    train_acc = sklearn.metrics.accuracy_score(y_train, y_pred)
    print("Mean acc (train): ", train_acc)

    idx_test_subset = get_target_indexes(dset_test, classes, 10)
    X_test = dset_test.data[idx_test_subset].type(torch.float)
    X_test = torch.unsqueeze(X_test, 1)
    y_test = dset_test.targets[idx_test_subset].numpy()

    model.eval()
    with torch.no_grad():
        mu, logvar=model.encode(X_test.to(model.device))
        z=model.reparameterize( mu, logvar)
        Z_test = z.cpu().numpy()

    y_pred = clf.predict(Z_test)
    test_acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    print("Mean acc (test): ", test_acc)

    cm = sklearn.metrics.confusion_matrix(y_true = y_test, y_pred = y_pred)
    df_cm = pd.DataFrame(cm, index = [str(i) for i in classes],
                           columns = [str(i) for i in classes])

    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plt.show()