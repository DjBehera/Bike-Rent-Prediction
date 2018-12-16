import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

#classifiaction.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#evaluation metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification


dataset  = pd.read_csv('train.csv')
dataset2  = pd.read_csv('test.csv')

##checking for the NA values
dataset.isnull().sum().sort_values()
dataset2.isnull().sum().sort_values()


#Converting the day into hourformat
dataset['hours'] = 0
dataset['months'] = 0
dataset['year'] = 0
for i in range(len(dataset)):
    dataset['hours'][i] = dataset['datetime'][i][11:13]
    dataset['months'][i] = dataset['datetime'][i][5:7]
    dataset['year'][i] = dataset['datetime'][i][:4]
    print('trainset:'+str(i))

dataset2['hours'] = 0
dataset2['months'] = 0
dataset2['year'] = 0
for i in range(len(dataset2)):
    dataset2['hours'][i] = dataset2['datetime'][i][11:13]
    dataset2['months'][i] = dataset2['datetime'][i][5:7]
    dataset2['year'][i] = dataset2['datetime'][i][:4]
    print('test_set:'+str(i))
#---------------------------------------------    

train_set = dataset.drop(['season','holiday','datetime','casual', 'registered','atemp'], axis = 1)
test_set = dataset2.drop(['season','holiday','datetime','atemp'],axis = 1)
#------------------------------------------------------------------------------------
y = train_set['count']
X = train_set.drop(['count'],axis = 1)
X_final = test_set
#-------------------------------------------------------------------------------------
categorical_list = ['workingday','hours','months','weather','year']


for i in categorical_list:
    temp_name = str(X[i][0])
    X = pd.get_dummies(X, columns=[i])
    X_final = pd.get_dummies(X_final, columns=[i])
    X = X.drop([i+'_%s' %(temp_name)],axis = 1)
    X_final = X_final.drop([i+'_%s' %(temp_name)],axis = 1) 
#-----------------------------------------------------------------------------------------    
normalized_list = ['temp','humidity','windspeed']

for i in normalized_list:
    X[i] = (X[i] - X[i].mean()) / (X[i].max() - X[i].min())
    X_final[i] = (X_final[i] - X_final[i].mean()) / (X_final[i].max() - X_final[i].min())
#--------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_final = sc.transform(X_final)   
#---------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
models=[RandomForestRegressor(),AdaBoostRegressor(),BaggingRegressor(),SVR(),KNeighborsRegressor()]

rmsle=[]

for model in range (len(models)):
    clf=models[model]
    print('fitting model %s' %model)
    clf.fit(X_train,y_train)
    test_pred=clf.predict(X_test)
    rmsle.append(np.sqrt(mean_squared_log_error(test_pred,y_test)))
    
regressor = RandomForestRegressor()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_final)

pred = pd.DataFrame({'datetime': dataset2['datetime'], 'count': y_pred})
pred.to_csv('bike_count.csv',index = False)