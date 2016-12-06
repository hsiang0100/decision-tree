import pandas as pd
import csv
import numpy as np
import math
from sklearn import tree
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import Imputer

#=====Load data and use dummies to process allegiances=====
df=pd.read_csv('character-deaths.csv', sep=',',header=None)
df.columns = ['Name','Allegiances','Death Year','Book of Death','Death Chapter','Book Intro Chapter','Gender','Nobility','GoT','CoK','SoS','FfC','DwD']
df_Allegiances = pd.get_dummies(df['Allegiances'])
df = pd.concat([df, df_Allegiances], axis=1)
df = df.drop('Book of Death',axis=1)
df = df.drop('Death Chapter',axis=1)

#=====Initial matrix=====
width, height = len(list(df.columns.values)), len(df)
Matrix = [[""for x in range(width)] for y in range(height)]
columns = ["Name", "Allegiances", "Death Year", "Book Intro Chapter", "Gender", "Nobility", "GoT", "CoK",
           "SoS", "FfC", "DwD", "Allegiances", "Arryn", "Baratheon", "Greyjoy", "House Arryn",
           "House Baratheon", "House Greyjoy", "House Lannister", "House Martell", "House Stark",
           "House Targaryen", "House Tully", "House Tyrell", "Lannister", "Martell", "Night's Watch",
           "None", "Stark", "Targaryen", "Tully", "Tyrell", "Wildling"]

#=====Tranfer data to matrix=====
#=====It can preprocess data in csv directly or in python
for j in range(0,33,1):
    Matrix[0][j] = columns[j]
for i in range(1,918,1):
    for j in range(0,33,1):
        if j==2:
            if np.isnan(float(df.values[i][2])):
                Matrix[i][2]=0
            else:
                Matrix[i][2]=1
                Matrix[i][2] = 1
        # ignore j=3 and 4
        elif j!=2:
            # if 0 no need other process
            if j==0 or j==1:
                Matrix[i][j] = df.values[i][j]
            # use j-2 to overlap
            else:
                if np.isnan(float(df.values[i][j])):
                    Matrix[i][j] = 0
                else:
                    Matrix[i][j] = df.values[i][j]

#=====Write data to csv=====
data = Matrix
f = open("after_process.csv",'w', newline='') 
w = csv.writer(f)
w.writerows(data)
f.close()

#=====Set X and y=====
train_df=pd.read_csv('after_process.csv')
#set feature
columns = ["Book Intro Chapter", "Gender", "Nobility", "GoT", "CoK",
           "SoS", "FfC", "DwD", "Arryn", "Baratheon", "Greyjoy", "House Arryn",
           "House Baratheon", "House Greyjoy", "House Lannister", "House Martell", "House Stark",
           "House Targaryen", "House Tully", "House Tyrell", "Lannister", "Martell", "Night's Watch",
           "None", "Stark", "Targaryen", "Tully", "Tyrell", "Wildling"]
features = train_df[list(columns)].values
y = targets = labels = train_df["Death Year"].values
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(features)

#=====Set train and test=====
trainnum = int(918*0.75)
testnum = int(918*0.75)+1
train_set_X = X[:trainnum]
train_set_y = y[:trainnum]
test_set_X = X[testnum:]
test_set_y = y[testnum:]

#=====Use fit and calculate score, precision and recall=====
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(train_set_X, train_set_y)
score = clf.score(test_set_X,test_set_y)
precision = precision_score(test_set_y, clf.predict(test_set_X))
recall = recall_score(test_set_y, clf.predict(test_set_X))
print ("score",score)
print ("precision",precision)
print ("recall",recall)

#=====Export graphviz to draw decision tree=====
with open("decision_tree.dot", 'w') as f:
  f = tree.export_graphviz(clf, out_file=f, feature_names=columns)