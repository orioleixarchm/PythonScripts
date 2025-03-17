#Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

pd.set_option('display.max_columns',None)

#Load data
path = Path("C:/Users/oriol/OneDrive/Programació/Python/Kaggle/Student Performance Factors")
file = "StudentPerformanceFactors.csv"
data = pd.read_csv(path / file)
print(data.head(),'\n')
print(data.dtypes,'\n')

for col in data.select_dtypes(include=['object','string']):
    print(f'Unique values of {col}: {data[col].nunique()}')
    print(data[col].unique())
    print('------------------------------------------------------------','\n')

data.dropna(inplace=True)

#Data preparation
Le = LabelEncoder()
Ss = StandardScaler()
Oe_Gen = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
Oe_Pif = OrdinalEncoder(categories=[['Negative','Neutral','Positive']])
Oe_Ped = OrdinalEncoder(categories=[['High School','College','Postgraduate']])
Oe_D = OrdinalEncoder(categories=[['Near','Moderate','Far']])
Le_columns = data.loc[:, data.nunique() == 2].columns
Ss_columns = data.select_dtypes(include='number').columns

for col in Le_columns:
    data[col] = Le.fit_transform(data[col])
data[Ss_columns] = Ss.fit_transform(data[Ss_columns])
data['Parental_Involvement'] = Oe_Gen.fit_transform(data[['Parental_Involvement']])
data['Access_to_Resources'] = Oe_Gen.fit_transform(data[['Access_to_Resources']])
data['Motivation_Level'] = Oe_Gen.fit_transform(data[['Motivation_Level']])
data['Family_Income'] = Oe_Gen.fit_transform(data[['Family_Income']])
data['Teacher_Quality'] = Oe_Gen.fit_transform(data[['Teacher_Quality']])
data['Peer_Influence'] = Oe_Pif.fit_transform(data[['Peer_Influence']])
data['Parental_Education_Level'] = Oe_Ped.fit_transform(data[['Parental_Education_Level']])
data['Distance_from_Home'] = Oe_D.fit_transform(data[['Distance_from_Home']])

#Variables an modle split
Y = data['Exam_Score']
X = data.drop(columns="Exam_Score")
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20, random_state=1)

#LassoCV
lasso = LassoCV(alphas=np.logspace(-4,4,100), cv=5, random_state=1)

#Regression Tree
tree = DecisionTreeRegressor(max_depth=8)

#Random Forest
rf = RandomForestRegressor(n_estimators=350, random_state=1)
rf.fit(xtrain, ytrain)
importance = rf.feature_importances_
print(pd.DataFrame({'Importance':importance}, index=xtrain.columns).sort_values(by='Importance', ascending=False).head(10),'\n')

#Models and metrics
models = {}
for model, name in zip([lasso, tree, rf], ['Lasso CV', 'Regression Tree', 'Random Forest']):
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    r2 = r2_score(ytest, ypred)
    mae = mean_absolute_error(ytest, ypred)
    mse = mean_squared_error(ytest, ypred)
    models[name] = [r2, mae, mse]

print(pd.DataFrame(models, index = ['R2', 'MAE', 'MSE']),'\n')
 