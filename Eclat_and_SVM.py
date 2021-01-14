# -*- coding: utf-8 -*-
###########################
# CSCI 573 Data Mining - Eclat and Linear Kernel SVM
# Author: Chu-An Tsai
# 12/14/2019
###########################

import fim 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

f = open('house-votes-84.data','r')
lines = f.readlines()
X = []
label = []
for line in lines:
    strpline = line.rstrip()
    arr = strpline.split(',')
    newline = [];
    for i in range(len(arr)):
        if arr[i] == 'y':
            newline.append(i)
    if arr[0] == 'republican':
        newline.append(100)
        label.append(0)
    else:
        newline.append(200)
        label.append(1)
    #print(*newline, sep=',')
    X.append(newline)

################################# a.
print('a. Run the itemset mining algorithm with 20% support. How many frequent itemsets are there?')
a = np.array(fim.eclat(X, supp=20))
print(len(a))  

################################# b.
b1 = fim.eclat(X, supp=20, report='a')
b2 = np.array(b1)
b3 = b2[b2[:,1].argsort()][::-1]
print('\nb. Write top 10 itemsets (in terms of highest support value).')
for i in range(10):
    print(b3[i])

################################# c.
print('\nc. How many frequent itemsets have 100 as part of itemsets?')
c1 = []
a=np.array(a)
for i in range(len(a)):
    if 100 in a[i][0]:
        c1.append(a[i].tolist())
c2 = np.array(c1)
c3 = c2[c2[:,1].argsort()][::-1].tolist()
print(len(c3))        

################################## d.
print('\nd. How many frequent itemsets have 200 as part of itemsets?')
d1 = []
for i in range(len(a)):
    if 200 in a[i][0]:
        d1.append(a[i].tolist())
d2 = np.array(d1)
d3 = d2[d2[:,1].argsort()][::-1].tolist()
print(len(d3))   
     
################################## e.
print('\ne. Write top 10 association rules (in terms of highest confidence value) where the rule''s head is 100.')
e1 = fim.eclat(X, supp=20, target='r', report='c', conf=75.0001)
e2 = np.array(e1)
e3 = e2[e2[:,2].argsort()][::-1]
e4 = []
for i in range(len(e3)):
    if e3[i][0] == 100:
        e4.append(e3[i].tolist())
e5 = np.array(e4)
for i in range(10):        
    print('confidence value:',e5[i][2],'    association rule:', e5[i][1], '→', e5[i][0],)

################################## f.
print('\nf. How many rules with head 100 are there for which the confidence value is more than 75%? List them.')
f1 = e5.copy()
count_100 = 0
for i in range(len(f1)):
    if (f1[i][2]) > 0.75:
        count_100 = count_100 + 1
        print('confidence value:', f1[i][2], '    association rule:', f1[i][1], '→', f1[i][0],)
print('Total:',count_100)

################################## g.
print('\ng. Write top 10 association rules (in terms of highest confidence value) where the rule''s head is 200.')
g2 = np.array(e1)
g3 = g2[g2[:,2].argsort()][::-1]
g4 = []
for i in range(len(g3)):
    if g3[i][0] == 200:
        g4.append(g3[i].tolist())
g5 = np.array(g4)
for i in range(10):        
    print('confidence value:',g5[i][2],'    association rule:', g5[i][1], '→', g5[i][0],)

################################## h.
print('\nh. How many rules with head 200 are there for which the confidence value is more than 75%? List them.')
h1 = g5.copy()
count_200 = 0
for i in range(len(h1)):
    if (h1[i][2]) > 0.75:
        count_200 = count_200 + 1
        print('confidence value:', h1[i][2], '    association rule:', h1[i][1], '→', h1[i][0],)
print('Total:',count_200)

################################### i.
print('\ni. soft-margin SVM with linear kernel')
i1 = e3[:,1].copy()
i2 = list(dict.fromkeys(i1))
i3 = np.zeros((len(X),len(i2))).astype(int)

for i in range(len(X)):
    for j in range(len(i2)):
        if (set(i2[j]).issubset(set(X[i]))) == True:
            i3[i][j] = 1
        else:
            i3[i][j] = 0

# Training set = first 75% data, Tuning set = 25% from training set, Test set = last 25% data 

data_train_lin_1, data_test_lin_1, data_train_label_lin_1, data_test_label_lin_1 = train_test_split(i3, label, train_size=0.75, random_state = 0, stratify = label)

#C = np.arange(0.01, 2, 0.01)
#parameters_linear = [{'C':C}]
parameters_linear = [{'C':[0.5, 0.7, 0.9, 1.0, 1.5]}]

model_linear = GridSearchCV(SVC(kernel='linear'), parameters_linear, cv=3).fit(data_train_lin_1, data_train_label_lin_1)
print('The best parameters: ', model_linear.best_params_)
#print("Scores for crossvalidation:")
#for mean, params in zip(model_linear.cv_results_['mean_test_score'], model_linear.cv_results_['params']):
    #print("Accuracy: %0.6f for %r" % (mean, params))
predicted_label_lin_1 =  model_linear.predict(data_test_lin_1)
accuracy_lin_1 = accuracy_score(data_test_label_lin_1, predicted_label_lin_1)
print('accurac:',accuracy_lin_1)

# Training set = last 75% data, Tuning set = 25% from training set, Test set = first 25% data 

data_test_lin_2, data_train_lin_2, data_test_label_lin_2, data_train_label_lin_2 = train_test_split(i3, label, train_size=0.25, random_state = 0, stratify = label)

model_linear = GridSearchCV(SVC(kernel='linear'), parameters_linear, cv=3).fit(data_train_lin_2, data_train_label_lin_2)
print('The best parameters: ', model_linear.best_params_)
#print("Scores for crossvalidation:")
#for mean, params in zip(model_linear.cv_results_['mean_test_score'], model_linear.cv_results_['params']):
    #print("Accuracy: %0.6f for %r" % (mean, params))
predicted_label_lin_2 =  model_linear.predict(data_test_lin_2)
accuracy_lin_2 = accuracy_score(data_test_label_lin_2, predicted_label_lin_2)
print('accurac:',accuracy_lin_2)

# Training set = first 37.5% and last 37.5%, Tuning set = 25% from training set, Test set = first 25% data 

data_temp1_lin_3, data_temp2_lin_3, data_temp1_label_lin_3, data_temp2_label_lin_3 = train_test_split(i3, label, train_size=0.375, random_state = 0, stratify = label)
data_test_lin_3, data_temp3_lin_3, data_test_label_lin_3, data_temp3_label_lin_3 = train_test_split(data_temp2_lin_3, data_temp2_label_lin_3, train_size=0.4, random_state = 0, stratify = data_temp2_label_lin_3)
data_train_lin_3 = np.vstack((data_temp1_lin_3, data_temp3_lin_3))
data_train_label_lin_3 = np.hstack((data_temp1_label_lin_3, data_temp3_label_lin_3))

model_linear = GridSearchCV(SVC(kernel='linear'), parameters_linear, cv=3).fit(data_train_lin_3, data_train_label_lin_3)
print('The best parameters: ', model_linear.best_params_)
#print("Scores for crossvalidation:")
#for mean, params in zip(model_linear.cv_results_['mean_test_score'], model_linear.cv_results_['params']):
    #print("Accuracy: %0.6f for %r" % (mean, params))
predicted_label_lin_3 =  model_linear.predict(data_test_lin_3)
accuracy_lin_3 = accuracy_score(data_test_label_lin_3, predicted_label_lin_3)
print('accurac:',accuracy_lin_3)

scores_lin = np.array([accuracy_lin_1, accuracy_lin_2, accuracy_lin_3])
print('Average 3-fold classification accuracy(along with standard deviation):', scores_lin.mean(), '(+/-',scores_lin.std(),')')


























