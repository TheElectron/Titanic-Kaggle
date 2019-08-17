# Importing libs.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def ages(data):
    data.Age = data.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 36, 60, 120)
    groups = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Y. Adult', 'Adult', 'Senior']
    categories = pd.cut(data.Age, bins, labels=groups)
    data.Age = categories
    return data


def cabins(data):
    data.Cabin = data.Cabin.fillna('N')
    data.Cabin = data.Cabin.apply(lambda x: x[0])
    return data


def fares(data):
    data.Fare = data.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    groups = ['Unknown', '1', '2', '3', '4']
    categories = pd.cut(data.Fare, bins, labels=groups)
    data.Fare = categories
    return data


def drops(data):
    return data.drop(['Ticket', 'Name', 'Embarked'], axis=1)


def transform_data(data):
    data = ages(data)
    data = cabins(data)
    data = fares(data)
    data = drops(data)
    return data


def encode(data_train, data_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(data_train[feature])
        data_train[feature] = le.transform(data_train[feature])
        le = le.fit(data_test[feature])
        data_test[feature] = le.transform(data_test[feature])
    return data_train, data_test


def run_kfold(classifier, name):
    kf = KFold(n_splits=5)
    outcomes = []
    fold = 0
    print("Classifier {0}".format(name))
    for train_index, test_index in kf.split(X):
        fold += 1
        x_train, x_test = X.values[train_index], X.values[test_index]
        y_train, y_test = Y.values[train_index], Y.values[test_index]
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome))
    print('\n')
    return  mean_outcome


def fiting_clf(clf, name):
        clf.fit(X_train, Y_train)
        score = clf.score(X_test, Y_test)
        print("Classifier {0} - Accuracy: {1}".format(name, score))


# Importing datasets.
DS_train = pd.read_csv('train.csv')
DS_train = transform_data(DS_train)
DS_test = pd.read_csv('test.csv')
DS_test = transform_data(DS_test)

# Data Vis.
sns.set(style="whitegrid")
sns.barplot(x="Age", y="Survived", data=DS_train)
plt.show()
sns.barplot(x="Fare", y="Survived", hue="Sex", data=DS_train)
plt.show()
sns.barplot(x="Cabin", y="Survived", hue="Sex", data=DS_train)
plt.show()
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=DS_train)
plt.show()

# Encoding data.
DS_train, DS_test = encode(DS_train, DS_test)

# Splitting up the Training Data.
X = DS_train.drop(['Survived', 'PassengerId'], axis=1)
Y = DS_train['Survived']
num_test = 0.20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=num_test, random_state=23)

# Choose the type of classifier.
names_classifiers = ['RandomForest', 'DecisionTree', 'AdaBoost', 'Gaussian', 'SVC']
classifiers = [RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               DecisionTreeClassifier(max_depth=5),
               AdaBoostClassifier(), GaussianNB(),
               SVC(gamma='auto')]

acurracy = []
for i in range(0, len(classifiers)):
    mean = run_kfold(classifiers[i], names_classifiers[i])
    acurracy.append(mean)

df = pd.DataFrame(dict(x=names_classifiers, y=mean))
sns.barplot(x="x", y="y", data=df)
plt.show()