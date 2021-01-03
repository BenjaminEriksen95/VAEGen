import sklearn
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from loaders import stratified_sampler
from loaders import get_target_indexes


def run_classifier(clf, model, dset_train, dset_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], batch_size=512):
    """Train, test and validate classifier clf

    The clf is initially trained with 100 labels from the training data and
    the accuracy is repoted. Then the clf is tested on the entire test set
    and accuracy is reported.
    Finally a confusion matrix is constructed and plotted
    """

    # training
    idx_train_subset = get_target_indexes(dset_train, classes, 10)
    X_train = dset_train.data[idx_train_subset].type(torch.float)
    # insert extra dim for the image channel
    X_train = torch.unsqueeze(X_train, 1)
    y_train = dset_train.targets[idx_train_subset].numpy()

    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(X_train.to(model.device))
        z = model.reparameterize(mu, logvar)
    Z_train = z.cpu().numpy()

    clf = clf
    clf.fit(Z_train, y_train)

    y_pred = clf.predict(Z_train)
    train_acc = sklearn.metrics.accuracy_score(y_train, y_pred)
    print("Mean acc (train): ", train_acc)

    # testing
    test_loader = torch.utils.data.DataLoader(dset_test, batch_size=batch_size,
                                              sampler=stratified_sampler(dset_test.targets, classes))

    test_acc_running = 0.
    cm = np.zeros(shape=(len(classes), len(classes)), dtype=np.int16)

    model.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            X_test = images.type(torch.float)
            y_test = labels.numpy()

            mu, logvar = model.encode(X_test.to(model.device))
            z = model.reparameterize(mu, logvar)
            Z_test = z.cpu().numpy()

            y_pred = clf.predict(Z_test)
            test_acc_batch = sklearn.metrics.accuracy_score(y_test, y_pred)
            test_acc_running += test_acc_batch

            cm_batch = sklearn.metrics.confusion_matrix(
                y_true=y_test, y_pred=y_pred)
            cm += cm_batch

    test_acc = test_acc_running / len(test_loader)
    print("Mean acc (test): ", test_acc)

    df_cm = pd.DataFrame(cm, index=[str(i) for i in classes],
                         columns=[str(i) for i in classes])

    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='d')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


def linearregression_classifier(model, dset_train, dset_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], batch_size=512):
    """Train, test and validate a LogisticRegression classifier"""
    clf = LogisticRegression(random_state=0)
    run_classifier(clf, model, dset_train, dset_test, classes, batch_size)


def kneighbors_classifier(model, dset_train, dset_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], batch_size=512):
    """Train, test and validate a KNeighborsClassifier"""
    clf = KNeighborsClassifier(n_neighbors=3)
    run_classifier(clf, model, dset_train, dset_test, classes, batch_size)


def mlp_classifier(model, dset_train, dset_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], batch_size=512):
    """Train, test and validate a MLPClassifier"""
    clf = MLPClassifier(hidden_layer_sizes=1500, max_iter=400, random_state=0)
    run_classifier(clf, model, dset_train, dset_test, classes, batch_size)
