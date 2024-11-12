#Loading packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#Loading and exploring data
path = 'C:/Users/oriol/OneDrive/Programació/Python/Kaggle/Salary'
file = 'Salary_Data'
filetype = 'csv'
data = pd.read_csv(f'{path}/{file}.{filetype}')
print(data.head(),'\n')
print(data.info(),'\n')
print('Unique Education Values:', data['Education Level'].unique(),'\n')
print('Unique Job Title Values:', data['Job Title'].unique(),'\n')

#Preparing data
data.dropna(inplace=True)
data.loc[data["Education Level"].isin(["Bachelor's Degree","Bachelor's"]),"Education Level"] = "Bachelor"
data.loc[data["Education Level"].isin(["Master's Degree","Master's"]),"Education Level"] = "Master"
data.loc[data["Education Level"].isin(["phd","phD"]),"Education Level"] = "PhD"

#Main statistics
print(data.describe(),'\n')
print(data[['Salary','Age','Years of Experience']].corr(),'\n')

#Visualization
data.groupby("Education Level")['Salary'].mean().sort_values().plot(kind='bar')
plt.title('Average Wage per Education Level')
plt.ylabel('Average Wage ($)')
plt.xticks(rotation=0)
plt.show()

sns.regplot(x=data['Years of Experience'], y=data['Salary'])
plt.title('Relation between Wage and Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Wage ($)')
plt.show()

sns.pairplot(data)
plt.show()

#Encoding
Le = LabelEncoder()
Oe = OrdinalEncoder(categories=[['High School',"Bachelor", "Master", "PhD"]])
Sc = StandardScaler()
data['Gender'] = Le.fit_transform(data['Gender'])
data['Education Level'] = Oe.fit_transform(data[['Education Level']])
data_encoded = pd.get_dummies(data, columns=['Job Title'], drop_first=True)
data_encoded[['Salary','Years of Experience']] = Sc.fit_transform(data_encoded[['Salary','Years of Experience']])
print(data_encoded.head(),'\n')

#Data Split
X = data_encoded.drop(columns='Salary')
Y = data_encoded['Salary']
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=1)

#Ordinary least Squares
OLS = LinearRegression()

#Lasso Regressor
lasso_CV = LassoCV(alphas=np.logspace(-4,4,100), cv=5, random_state=1)

#Tree Regressor
tree = DecisionTreeRegressor(max_depth=6)

#Random Forest
rforest = RandomForestRegressor(n_estimators=500, random_state=1)

#Accuracy
models = {}
regressions = [OLS, lasso_CV, tree, rforest]
model_names = ['OLS', 'Lasso CV', 'Regression Tree', 'Random Forest']
for model,name in zip(regressions,model_names):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = round(mean_squared_error(y_test, y_pred),3)
    r_squared = round(r2_score(y_test, y_pred),3)
    print(f'{name} MSE: {mse}, R2: {r_squared}', '\n')
    models[name] = [mse, r_squared]

print(pd.DataFrame(models, index=['MSE', 'R2']).head())

