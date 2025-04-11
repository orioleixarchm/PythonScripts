#Loading packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

pd.set_option('display.max_columns', None)

#Loading data
path = "C:/Users/oriol/OneDrive/Programació/Python/Kaggle/Student Dropout"
file = "Predict Student Dropout and Academic Success"
type = "csv"
data = pd.read_csv(f'{path}/{file}.{type}', sep=';')
print(data.head(), '\n')
print(data.columns, '\n')
print(f'Target column has {data['Target'].nunique()} values: {data['Target'].unique()}', '\n')

data_sub = data.loc[data['Target']!='Unknown',['Marital status','Daytime/evening attendance\t', 'Admission grade',
       'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 
       'Gender', 'Scholarship holder', 'Age at enrollment', 'International',
       'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
       'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
       'Inflation rate', 'GDP', 'Target']].dropna()

print(data_sub.dtypes, '\n')

#Data preparation
Le = LabelEncoder()
Se = StandardScaler()

le_columns = ['Marital status','Daytime/evening attendance\t','Displaced', 'Educational special needs', 'Debtor', 'Gender', 'Scholarship holder', 'International','Target']
for col in le_columns:
    data_sub[col] = Le.fit_transform(data_sub[col])
se_columns = data_sub.drop(columns=le_columns).columns
data_sub[se_columns] = Se.fit_transform(data_sub[se_columns])

#Model Split
Y = data_sub['Target']
X = data_sub.drop(columns='Target')
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=1)

#Logit
logit = LogisticRegression()

#Decision Tree
tree = DecisionTreeClassifier(max_depth=8)

#Random Forest
rf = RandomForestClassifier(n_estimators=300, random_state=1)
rf.fit(xtrain, ytrain)
importance = rf.feature_importances_
print(pd.DataFrame({'Importance': importance}, index=xtrain.columns).sort_values(by='Importance', ascending=False).head())

#Extreme Gradient Boost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

#Metrics and models
models = {}
for model, name in zip([logit, tree, rf, xgb_model], ['Logistic', 'Decision Tree', 'random Forest', 'Extreme Gradient Boost']):
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    acc = accuracy_score(ytest, ypred)
    pre = precision_score(ytest, ypred, average='macro')
    cm = confusion_matrix(ytest, ypred)
    ConfusionMatrixDisplay(cm, display_labels=['Dropout','Graduate','Enrolled']).plot(cmap='Greens')
    plt.title(f'Confusion Matrix of the {name} model with an accuracy of {round(acc,3)}')
    plt.show()
    print(f'Classification report of the {name} model:', '\n')
    print(classification_report(ytest, ypred), '\n')
    models[name] = [acc, pre]

print(pd.DataFrame(models, index=['ACC', 'PRE']))
