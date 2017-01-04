
# coding: utf-8

# In[1]:

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interp
from itertools import cycle
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')


# In[2]:

colors = cycle(['brown','lightcoral','red','magenta','cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])


# In[3]:

def algorithm(algoname,colors,train,test,pos):
    mean_tpr,lw,i =0.0, 2,1
    mean_fpr = np.linspace(0, 1, 100)
    fold_accuracy= []
    cnf_mat = 0
    skfold = StratifiedKFold(n_splits=10,shuffle = True)
    for (trainindex,testindex), color in zip(skfold.split(train, test.values.ravel()), colors):
        X_train, X_test = train.loc[trainindex], train.loc[testindex]
        y_train, y_test = test.loc[trainindex], test.loc[testindex]
        model = algoname.fit(X_train,y_train.values.ravel())
        fold_accuracy.append(model.score(X_test,y_test.values.ravel()))
        result = model.predict(X_test)
        fpr, tpr, thresholds= roc_curve(y_test.values,result,pos_label=pos)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_test.values,result)
        cnf_mat +=  cm
        plt.step(fpr, tpr, lw=lw, color=color,label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        i+=1
    mean_tpr /= skfold.get_n_splits(train,test.values.ravel())
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.step(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
    plt.title("Average accuracy: {0:.3f}".format(np.asarray(fold_accuracy).mean()))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right") 
    plt.show()
    return("Average accuracy: {0:.3f} (+/-{1:.3f})".format(np.asarray(fold_accuracy).mean(),                                                            np.asarray(fold_accuracy).std()),           "\n Confustion Matrix:",cnf_mat)   


# In[4]:

german = pd.read_csv('german.data',header = None, sep = " ")
features = ['Checking account','Duration(month)','Credit history','Purpose',           'Credit Amount','Savings/Stocks','Present employment Length',           'Installment rate','Personal status','Guarantors',           'Residing since','Property','Age(years)','Other installment plans',           'Housing','No of credits',           'Job','dependents','Telephone','foreign worker']
label = ['Creditability']
columns = features + label
german.columns = columns
for col in german.columns.values:
    if german[col].dtype == 'object':
        german[col] = LabelEncoder().fit_transform(german[col])
german_train,german_test = german[features],german[label]


# In[5]:

default = pd.read_csv('default of credit card clients.csv')
default=default.rename(columns = {'default payment next month':'default'})
default_train,default_test = default.iloc[:,:len(default.columns)-1],default.iloc[:,len(default.columns)-1]


# In[6]:

crx = pd.read_csv('crx.data',header=None,sep = ',')
cols = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15']
classlabel = ['A16']
columns = cols + classlabel
crx.columns = columns
for col in crx.columns.values:
    if crx[col].dtype == 'object':
        crx[col] = LabelEncoder().fit_transform(crx[col])
crx_train, crx_test = crx[cols],crx[classlabel]


# Random Forest

# In[7]:
print("\nRandom Forest")
print("\n German Dataset")
forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=800, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=50,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=600, n_jobs=-1, oob_score=False,
            random_state=None, verbose=0, warm_start=False)
print(algorithm(forest,colors,german_train,german_test,pos = 2))


# In[8]:
print("\n Credit Approval Data Set")
forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=300, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=800, n_jobs=-1, oob_score=False, random_state=100,
            verbose=0, warm_start=False)
print(algorithm(forest,colors,crx_train,crx_test,pos = None))


# In[9]:
print("\n Default of Credit Card Clients Data Set")
forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=100, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=50,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=600, n_jobs=-1, oob_score=False,
            random_state=None, verbose=0, warm_start=False)
print(algorithm(forest,colors,default_train,default_test,pos = None))


# Logistic Regression

# In[10]:
print("\nLogistic Regression")
print("\n German Dataset")
logistic = LogisticRegression(C=0.5, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=-1,
          penalty='l2', random_state=500, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
print(algorithm(logistic,colors,german_train,german_test,pos = 2))


# In[11]:
print("\n Credit Approval Data Set")
logistic = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=-1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
print(algorithm(logistic,colors,crx_train,crx_test,pos = None))


# In[12]:
print("\n Default of Credit Card Clients Data Set")
logistic = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=-1,
          penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
print(algorithm(logistic,colors,default_train,default_test,pos = None))


# Naive Bayes

# In[13]:
print("\n Naive Bayes") 
print("\n German Dataset")
naive = GaussianNB()
print(algorithm(naive,colors,german_train,german_test,pos = 2))


# In[14]:
print("\n Credit Approval Data Set")
naive = GaussianNB()
print(algorithm(naive,colors,crx_train,crx_test,pos = None))


# In[15]:
print("\n Default of Credit Card Clients Data Set")
naive = GaussianNB()
print(algorithm(naive,colors,default_train,default_test,pos = None))


# k Nearest Neighbors

# In[16]:
print("\n k Nearest Neighbors")    
print("\n German Dataset")
knneigh = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
           weights='uniform')
print(algorithm(knneigh,colors,german_train,german_test,pos = 2))


# In[17]:
print("\n Credit Approval Data Set")
knneigh = KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=-1, n_neighbors=10, p=2,
           weights='uniform')
print(algorithm(knneigh,colors,crx_train,crx_test,pos = None))


# In[18]:
print("\n Default of Credit Card Clients Data Set")
knneigh = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=-1, n_neighbors=50, p=2,
           weights='uniform')
print(algorithm(knneigh,colors,default_train,default_test,pos = None))


# Support Vector Machines

# In[19]:
print("\n Support Vector Machines")
    
print("\n German Dataset")
svm = LinearSVC(C=1, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l1', random_state=1000, tol=0.0001,
     verbose=0)
print(algorithm(svm,colors,german_train,german_test,pos = 2))


# In[20]:
print("\n Credit Approval Data Set")
svm = LinearSVC(C=1, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=500,
     multi_class='ovr', penalty='l2', random_state=1000, tol=0.0001,
     verbose=0)
print(algorithm(svm,colors,crx_train,crx_test,pos = None))


# In[21]:
print("\n Default of Credit Card Clients Data Set")
svm = LinearSVC(C=1, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=10,
     multi_class='ovr', penalty='l1', random_state=1000, tol=0.0001,
     verbose=0)
print(algorithm(svm,colors,default_train,default_test,pos = None))


# In[ ]:



