#Loading packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, precision_score, ConfusionMatrixDisplay, classification_report

pd.set_option('display.max_columns',None)

#Loading data
path = 'C:/Users/oriol/OneDrive/Programació/Python/Kaggle/Credit Risk'
file = 'credit'
type = 'csv'
data = pd.read_csv(f'{path}/{file}.{type}')
print(data.head(),'\n')

#Data exploration
print(f'Nº rows: {data.shape[0]}','\n')
for column in ['employment_duration','job','other_credit','credit_history','checking_balance','purpose','housing',
               'employment_duration','savings_balance', 'default']:
    print(f'Unique values of {column}: {data[column].nunique()}')
    print('---------------------------------------')
    print(data[column].unique(),'\n')

#Data preparation
variables = ['checking_balance', 'months_loan_duration', 'credit_history', 'purpose', 'amount', 'savings_balance',
              'employment_duration', 'age', 'housing', 'existing_loans_count', 'job', 'default']
data_enc = data.loc[:, variables]
data_enc.loc[data_enc['purpose']=='car0','purpose'] = 'car'
data_enc['checking_balance'] = data_enc['checking_balance'].replace('unknown', data_enc['checking_balance'].value_counts().index[1])
data_enc['savings_balance'] = data_enc['savings_balance'].replace('unknown', data_enc['savings_balance'].mode()[0])
print(f'Nº rows now: {data_enc.shape[0]}','\n')
Le = LabelEncoder()
OeCh = OrdinalEncoder(categories=[['critical','poor','good','very good', 'perfect']])
OeC = OrdinalEncoder(categories=[['< 0 DM','1 - 200 DM','> 200 DM']])
OeS = OrdinalEncoder(categories=[['< 100 DM', '100 - 500 DM', '500 - 1000 DM', '> 1000 DM']])
OeE = OrdinalEncoder(categories=[['unemployed','< 1 year','1 - 4 years','4 - 7 years','> 7 years']])
Ss = StandardScaler()
data_enc['default'] = Le.fit_transform(data_enc['default'])
data_enc['credit_history'] = OeCh.fit_transform(data_enc[['credit_history']])
data_enc['checking_balance'] = OeC.fit_transform(data_enc[['checking_balance']])
data_enc['savings_balance'] = OeS.fit_transform(data_enc[['savings_balance']])
data_enc['employment_duration'] = OeE.fit_transform(data_enc[['employment_duration']])
data_enc[['amount','months_loan_duration','age','existing_loans_count']] = Ss.fit_transform(data_enc[['amount','months_loan_duration','age','existing_loans_count']])
data_enc = pd.get_dummies(data=data_enc, columns=['purpose','housing','job'], drop_first=True)

#Data Split
Y = data_enc['default']
X = data_enc.drop(columns='default')
xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.2, random_state=1)

#Logistic Reression
logit = LogisticRegression()

#Decision tree
tree = DecisionTreeClassifier(max_depth=7)

#Random Forest
rf = RandomForestClassifier(n_estimators=250, random_state=1)
rf.fit(xtrain,ytrain)
importances = rf.feature_importances_
print("Most important features:","\n",pd.DataFrame({'Importance':importances}, index=xtest.columns).sort_values(by='Importance', ascending=False).head(),'\n')

#Extreme Gradient Boost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

#Model accuracy and results
models = {}
for model, name in zip([logit, tree, rf, xgb_model], ['Logistic Regression', 'Classification Tree', 'Random Forest', 'Xtreme Gradient Boost']):
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    acc = accuracy_score(ytest, ypred)
    mae = mean_absolute_error(ytest, ypred)
    mse = mean_squared_error(ytest, ypred)
    ps = precision_score(ytest, ypred)
    print(f'Report of the {name} model:')
    print(classification_report(ytest, ypred))
    cm = confusion_matrix(ytest,ypred)
    ConfusionMatrixDisplay(cm, display_labels=['Default', 'No Default']).plot(cmap='Blues')
    plt.title(f'Confusion Matrix of the {name} model:')
    plt.show()
    models[name] =[acc, mae, mse, ps]

print(pd.DataFrame(models,index=['Accuracy','MAE','MSE','Precision']))
