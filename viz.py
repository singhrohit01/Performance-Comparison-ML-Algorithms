import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os


#Reading all the datasets.
datasets = sorted(os.listdir("python_data"))
num_datasets = len(datasets)

for data in datasets:
    train_df = pd.read_csv('python_data/' + data)
    label_name = train_df.columns[-1]
    train_df = pd.get_dummies(train_df)
    train_df.fillna(-999, inplace=True)
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        min_samples_leaf=4,
        max_features=0.5)
    rf.fit(train_df.drop([label_name], axis=1), train_df[label_name])
    old_features = list(train_df.drop([label_name], axis=1))
    feature_importances = {}
    feature_importances_list = rf.feature_importances_
    for index, feature in enumerate(old_features):
        feature_importances[feature] = feature_importances_list[index]
    itemlist = sorted(feature_importances.items(), key=operator.itemgetter(1), reverse=True)

    #Here we plot the 20 most important features.
    fig, ax = plt.subplots()
    feature_names = [x[0] for x in itemlist[:20]]
    feature_importance = [x[1] for x in itemlist[:20]]

    ax.barh(feature_names, feature_importance, align='center')
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')

    #Here we visualize the top 4 features.
    plt.show()
    feature_to_show = feature_names[:4]
    g = sns.pairplot(train_df, vars=feature_to_show, hue=label_name)
