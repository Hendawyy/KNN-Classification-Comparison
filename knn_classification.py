import pandas as pd
import numpy as np

CRX = pd.read_csv('crxdataReducedClean.csv',  names=["col0","col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9"])

#Check if There is Any Null Value in The Dataset.
print(CRX.isnull().sum())

from sklearn.preprocessing import LabelEncoder
#Encoding categorical data.
CRX[['col2', 'col3', 'col4']] = CRX[['col2', 'col3', 'col4']].apply(LabelEncoder().fit_transform)
CRX.dtypes
CRX['col7'] = pd.to_numeric(CRX['col7'],errors='coerce')
CRX.dtypes
print(CRX.isnull().sum())
#Remove Any Null Value in The Dataset.
CRX=CRX.dropna()
print(CRX.isnull().sum())
CRX.dtypes
X = CRX.iloc[:, 0:9].values
Y = CRX.iloc[:, 9].values

#Using the Train_test_split to Split dataset into Train and test.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)



'''''''''''''''''''''''''''''''''''''''''''''''''''No Normalization'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


#Optimal Value Of K
acc = []

for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(x_train,y_train)
    yhat = neigh.predict(x_test)
    acc.append(metrics.accuracy_score(y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
K=acc.index(max(acc))
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))

#KNN with K = Optimum Using The Distance Equation Minkowski.
ClassifierM = KNeighborsClassifier(n_neighbors=K, metric ='minkowski', p=2)
ClassifierM.fit(x_train, y_train)
y_pred = ClassifierM.predict(x_test)


#Confusion Matrix For K = Optimum Using The Distance Equation Minkowski.
k5cfM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix : ')
print(k5cfM)

#Classification Report For K = Optimum Using The Distance Equation Minkowski.
k5crM = classification_report(y_test, y_pred)
print('Classification Report : ')
print(k5crM)

#Accuracy Score For K = Optimum Using The Distance Equation Minkowski.
k5asM = accuracy_score(y_test, y_pred)
print('Accuracy :',k5asM*100,'%')



#KNN with K = Optimum Using The Distance Equation Euclidean.
ClassifierE = KNeighborsClassifier(n_neighbors=K, metric ='euclidean')
ClassifierE.fit(x_train, y_train)
y_predE5 = ClassifierE.predict(x_test)


#Confusion Matrix For K = Optimum Using The Distance Equation Euclidean.
k5cfE = confusion_matrix(y_test, y_predE5)
print('Confusion Matrix : ')
print(k5cfE)

#Classification Report For K = Optimum Using The Distance Equation Euclidean.
k5crE = classification_report(y_test, y_predE5)
print('Classification Report : ')
print(k5crE)

#Accuracy Score For K = Optimum Using The Distance Equation Euclidean.
k5asE = accuracy_score(y_test, y_predE5)
print('Accuracy :',k5asE*100,'%')



#KNN with K = Optimum Using The Distance Equation Manhattan.
ClassifierN = KNeighborsClassifier(n_neighbors=K, metric ='manhattan')
ClassifierN.fit(x_train, y_train)
y_predN5 = ClassifierN.predict(x_test)


#Confusion Matrix For K = Optimum Using The Distance Equation Manhattan.
k5cfN = confusion_matrix(y_test, y_predN5)
print('Confusion Matrix : ')
print(k5cfN)

#Classification Report For K = Optimum Using The Distance Equation Manhattan.
k5crN = classification_report(y_test, y_predN5)
print('Classification Report : ')
print(k5crN)

#Accuracy Score For K = Optimum Using The Distance Equation Manhattan.
k5asN = accuracy_score(y_test, y_predN5)
print('Accuracy :',k5asN*100,'%')


#Intialize A Dataframe to compare results from the used distance equations. 
K5C = {'Confusion Matrix' : pd.Series([k5cfM, k5cfE, k5cfN], index =['Minkowski', 'Euclidean', 'Manhattan']),
      'Accuracy Score' : pd.Series([k5asM, k5asE, k5asN], index =['Minkowski', 'Euclidean', 'Manhattan'])} 

#Create an Comparison Dataframe That Holds a comparison between The Accuracy Scores and the confusion matrices of the diffrent Distance Functions. 
K5 = pd.DataFrame(K5C) 

print('K = Optimum\n',K5)








'''''''''''''''''''''''''''''''''''''''''''''''''''After Normalization'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#Using Feature Scaling to Scale Values in X To Numerical Values Between -1.5 & 1.5.
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

#Optimal Value Of K
acc = []

for i in range(1,40):
    neigha = KNeighborsClassifier(n_neighbors = i).fit(x_train,y_train)
    yhata = neigha.predict(x_test)
    acc.append(metrics.accuracy_score(y_test, yhata))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
Ka=acc.index(max(acc))
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))

#KNN with K = Optimum Using The Distance Equation Minkowski.
ClassifierMp = KNeighborsClassifier(n_neighbors=Ka, metric ='minkowski', p=2)
ClassifierMp.fit(x_train, y_train)
y_predp = ClassifierMp.predict(x_test)


#Confusion Matrix For K = Optimum Using The Distance Equation Minkowski.
k5cfMp = confusion_matrix(y_test, y_predp)
print('Confusion Matrix : ')
print(k5cfMp)

#Classification Report For K = Optimum Using The Distance Equation Minkowski.
k5crMp = classification_report(y_test, y_predp)
print('Classification Report : ')
print(k5crMp)

#Accuracy Score For K = Optimum Using The Distance Equation Minkowski.
k5asMp = accuracy_score(y_test, y_predp)
print('Accuracy :',k5asMp*100,'%')



#KNN with K = Optimum Using The Distance Equation Euclidean.
ClassifierEp = KNeighborsClassifier(n_neighbors=Ka, metric ='euclidean')
ClassifierEp.fit(x_train, y_train)
y_predE5p = ClassifierEp.predict(x_test)


#Confusion Matrix For K = Optimum Using The Distance Equation Euclidean.
k5cfEp = confusion_matrix(y_test, y_predE5p)
print('Confusion Matrix : ')
print(k5cfEp)

#Classification Report For K = Optimum Using The Distance Equation Euclidean.
k5crEp = classification_report(y_test, y_predE5p)
print('Classification Report : ')
print(k5crEp)

#Accuracy Score For K = Optimum Using The Distance Equation Euclidean.
k5asEp = accuracy_score(y_test, y_predE5p)
print('Accuracy :',k5asEp*100,'%')



#KNN with K = Optimum Using The Distance Equation Manhattan.
ClassifierNp = KNeighborsClassifier(n_neighbors=Ka, metric ='manhattan')
ClassifierNp.fit(x_train, y_train)
y_predN5p = ClassifierNp.predict(x_test)


#Confusion Matrix For K = Optimum Using The Distance Equation Manhattan.
k5cfNp = confusion_matrix(y_test, y_predN5p)
print('Confusion Matrix : ')
print(k5cfNp)

#Classification Report For K = Optimum Using The Distance Equation Manhattan.
k5crNp = classification_report(y_test, y_predN5p)
print('Classification Report : ')
print(k5crNp)

#Accuracy Score For K = Optimum Using The Distance Equation Manhattan.
k5asNp = accuracy_score(y_test, y_predN5p)
print('Accuracy :',k5asNp*100,'%')


#Intialize Data to be Put in The Dafarame. 
K5Cp = {'Confusion Matrix' : pd.Series([k5cfMp, k5cfEp, k5cfNp], index =['Minkowski', 'Euclidean', 'Manhattan']),
      'Accuracy Score' : pd.Series([k5asMp, k5asEp, k5asNp], index =['Minkowski', 'Euclidean', 'Manhattan'])} 

#Create an Comparison Dataframe That Holds a comparison between The Accuracy Scores and the confusion matrices of the diffrent Distance Functions. 
K5p = pd.DataFrame(K5Cp) 

print('K = Optimum\n',K5p)


FPDF = pd.concat([K5, K5p], axis = 1)
a=pd.DataFrame(FPDF)
