import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression


def skc(X_train_mat, y_train_vec, network):
    #Here we choose the model.
    if network == 'lr':
        clf = LogisticRegression()
    elif network == 'rf':
        clf = RandomForestClassifier()
    elif network == 'svm':
        clf = SVC()
    elif network == 'knn':
        clf = KNeighborsClassifier()

    # Here we use K-fold cross validation. 
    scores = cross_validate(
        clf,
        X_train_mat,
        y_train_vec,
        scoring=[
            'accuracy',
            'f1',
            'roc_auc'],
        cv=10,
        return_train_score='True')
    train_accuracy = scores['train_accuracy'].mean()
    test_accuracy = scores['test_accuracy'].mean()
    train_f1 = scores['train_f1'].mean()
    test_f1 = scores['test_f1'].mean()
    train_auc = scores['train_roc_auc'].mean()
    test_auc = scores['test_roc_auc'].mean()
    return train_accuracy, test_accuracy, train_f1, test_f1, train_auc, test_auc


def preprocess(dataset, should_scale):
    #Here we preprocess the data.
    data = pd.read_csv('python_data/' + dataset)
    heads = list(data.columns.values)
    rows, cols = data.shape
    X = data.iloc[:, :cols - 1]
    y = data.iloc[:, cols - 1]
    data_types = X.dtypes
    categorical_data = []
    numeric_data = []
    #Convert categorical data into one hot vecotrs.
    for col_type, col in zip(data_types, X.columns):
        if col_type == 'object' or col_type == 'bool':
            categorical_data.append(col)
        elif col_type == 'int64' or col_type == 'float64':
            numeric_data.append(col)
    #Mean imputation for missing values.
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_mean.fit(data)
    data = imp_mean.transform(data)
    data = pd.DataFrame(data)
    data.columns = heads
    X = data.iloc[:, :cols - 1]
    y = data.iloc[:, cols - 1]
    y = y.astype('int')
    #Scaling the values.
    if should_scale == 'scale':
        scaler = MinMaxScaler(feature_range=(0.001, 0.999))
        X[numeric_data] = scaler.fit_transform(X[numeric_data])
        X = pd.DataFrame(X)
    X = pd.get_dummies(X, columns=categorical_data)

    return X, y


def forward(X_train, y_train):
    #Here we calculate the scores and plot the graphs.
    train_accuracy = []
    test_accuracy = []
    train_f1 = []
    test_f1 = []
    train_auc = []
    test_auc = []

    for network in ['lr', 'rf', 'svm', 'knn']:
        new_train_acc, new_test_acc, new_train_f1, new_test_f1, new_train_auc, new_test_auc = skc(
            X_train, y_train, network)

        train_accuracy.append(new_train_acc)
        test_accuracy.append(new_test_acc)

        train_f1.append(new_train_f1)
        test_f1.append(new_test_f1)

        train_auc.append(new_train_auc)
        test_auc.append(new_test_auc)

    #Here we plot the graphs and save them.
    x = np.array([0, 1, 2, 3])
    y = np.array(train_accuracy)
    y2 = np.array(test_accuracy)
    plt.plot(x, y, 'g^', label='Training Accuracy')
    plt.plot(x, y2, 'bs', label='Validation Accuracy')
    my_xticks = ['lr', 'rf', 'svm', 'knn']
    plt.xticks(x, my_xticks)
    plt.ylabel('Accuracy')
    plt.xlabel('Method')
    plt.savefig('%s_%s_%s.png' % (dataset, 'Accuracy', should_scale))
    plt.clf()

    x = np.array([0, 1, 2, 3])
    y = np.array(train_f1)
    y2 = np.array(test_f1)
    plt.plot(x, y, 'g^', label='Training F1 score')
    plt.plot(x, y2, 'bs', label='Validation F1 score')
    my_xticks = ['lr', 'rf', 'svm', 'knn']
    plt.xticks(x, my_xticks)
    plt.ylabel('F1 Score')
    plt.xlabel('Method')
    plt.savefig('%s_%s_%s.png' % (dataset, 'f1_score', should_scale))
    plt.clf()

    x = np.array([0, 1, 2, 3])
    y = np.array(train_auc)
    y2 = np.array(test_auc)
    plt.plot(x, y, 'g^', label='Training AUC')
    plt.plot(x, y2, 'bs', label='Validation AUC')
    my_xticks = ['lr', 'rf', 'svm', 'knn']
    plt.xticks(x, my_xticks)
    plt.ylabel('AUC')
    plt.xlabel('Method')
    plt.savefig('%s_%s_%s.png' % (dataset, 'AUC', should_scale))
    plt.clf()

#Reading all the datasets.
datasets = sorted(os.listdir("python_data"))
num_datasets = len(datasets)


do_scale = ['scale', 'no_scale']
for dataset in datasets:
    for should_scale in do_scale:
        X, y = preprocess(dataset, should_scale)
        forward(X, y)
