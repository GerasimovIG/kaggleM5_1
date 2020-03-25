from sklearn.datasets import make_circles
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
        
sell_prices = pd.read_csv(r'C:\Users\Ilia\Documents\git\kaggleM5_data\sell_prices.csv', engine='python')
calendar = pd.read_csv(r'C:\Users\Ilia\Documents\git\kaggleM5_data\calendar.csv', engine='python')
submission_format = pd.read_csv(r'C:\Users\Ilia\Documents\git\kaggleM5_data\sample_submission.csv', engine='python')
train = pd.read_csv(r'C:\Users\Ilia\Documents\git\kaggleM5_data\sales_train_validation.csv', engine='python')



train = train.drop(labels = ["Name", "Ticket", "Cabin"],axis = 1)
train['Sex'] = train['Sex'].replace('male', '1').replace('female', '0')
train["Sex"] = train["Sex"].astype(int)

for i in train.columns:
    if train[i].dtypes == np.object:
        train = pd.concat([train.drop(i, axis=1), pd.get_dummies(train[i], prefix=[i])], axis=1)

train = train.dropna()
train.describe(include = 'all')


train.isnull().sum() #проверка на наличие нулов в каждом столбце (подробнее https://chartio.com/resources/tutorials/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe/)

y = train["Survived"].values
X = train.drop(labels = ["Survived", "PassengerId"],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier = RandomForestClassifier(n_estimators=200, random_state=0)  
classifier.fit(X_train, y_train)
predictions = pd.DataFrame(data=classifier.predict(X_test))  
predictions2 = pd.DataFrame(data=classifier.predict_proba(X_test))

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, predictions)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, predictions)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, predictions)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, predictions)
print('F1 score: %f' % f1)

test = test.drop(labels = ["Name", "Ticket", "Cabin"],axis = 1)
test['Sex'] = test['Sex'].replace('male', '1').replace('female', '0')
test["Sex"] = test["Sex"].astype(int)

for i in test.columns:
    if test[i].dtypes == np.object:
        test = pd.concat([test.drop(i, axis=1), pd.get_dummies(test[i], prefix=[i])], axis=1)

for (columnName, columnData) in test.iteritems():#заменить нулы на среднее
    test[columnName].fillna((test[columnName].mean()), inplace=True)
    #print('Colunm Name : ', columnName)
    #print('Column Contents : ', columnData.values)

test.isnull().sum() #проверка на наличие нулов в каждом столбце (подробнее https://chartio.com/resources/tutorials/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe/)


classifier.fit(X, y)
predictions = pd.DataFrame(data=classifier.predict(test.drop(labels = ["PassengerId"],axis = 1)))  
predictions['PassengerId'] = test['PassengerId']
predictions[['PassengerId', 'Survived']] = predictions[['PassengerId', 0]]
predictions = predictions.drop(labels = [0],axis = 1)
predictions.to_csv('predictionsTest.csv', index=False)
predictions.describe(include = 'all')
predictions.head()

#whos
#reset
