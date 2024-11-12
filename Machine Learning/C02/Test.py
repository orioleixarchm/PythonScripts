#Loading Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression

#Loading data
path = "C:/Users/oriol/Downloads"
file = 'climate_change_impact_on_agriculture_2024.csv'
data = pd.read_csv(f'{path}/{file}')
print('Data:')
print(data.head())
print('\n')
print('Columns in the dataset:')
print(data.columns)
print('\n')
print('Data information:')
print(data.info())
print('\n')
print('Unique values for relevant columns:')
for column in ['Region', 'Country', 'Crop_Type', 'Adaptation_Strategies']:
    print(f'{column}:')
    print('------------------------------------')
    print(data[column].unique())
    print('\n')

#Data Cleaning
print('Nulls per column:')
print(data.isnull().sum())
print('\n')

#Extreme event Plot
extreme_events = data.groupby(['Year','Country'])['Extreme_Weather_Events'].size().reset_index(name='Events')
ex = sns.FacetGrid(extreme_events, col='Country', col_wrap=5)
ex.map(sns.lineplot, 'Year', 'Events')
ex.set_titles('Country: {col_name}')
ex.set_xlabels('Year')
ex.set_ylabels('Nº Events')

#Economic impact of Adaptation strategy
strategies_total = data.groupby('Adaptation_Strategies')['Economic_Impact_Million_USD'].sum()
strategies_total.sort_values(ascending=False, inplace=True)
order = strategies_total.index
print('Most profitable strategies:')
print(strategies_total)
print('\n')

strategies = data.groupby(['Adaptation_Strategies','Country'])['Economic_Impact_Million_USD'].sum().reset_index()
s = sns.FacetGrid(strategies, col='Country', col_wrap=5)
s.map(sns.barplot,'Economic_Impact_Million_USD','Adaptation_Strategies', order=order)
s.set_titles('Country: {col_name}')
s.set_xlabels('Economic Impact MUSD')
s.set_ylabels('Strategy')


#Correlation
##corr_matric = data.select_dtypes(include='number')
corr_matrix = data[['Average_Temperature_C','Total_Precipitation_mm','CO2_Emissions_MT','Crop_Yield_MT_per_HA'
            ,'Extreme_Weather_Events','Irrigation_Access_%','Pesticide_Use_KG_per_HA','Fertilizer_Use_KG_per_HA'
            ,'Soil_Health_Index','Economic_Impact_Million_USD']].corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr_matrix, fmt='.2f', annot=True)
plt.xticks(rotation=20)


#Geographical Impact
top_3_impact = data.groupby('Country')['Economic_Impact_Million_USD'].mean().nlargest(3).reset_index()
plt.figure(figsize=(8,8))
sns.barplot(top_3_impact, x='Country',y='Economic_Impact_Million_USD')
plt.xlabel(None)
plt.ylabel('Impact MUSD')
plt.title('Top 3 Countries per USD Impact')
plt.show()

#RF
data_dummies = pd.get_dummies(data, columns=['Region', 'Crop_Type','Adaptation_Strategies'])
##data_dummies[data_dummies.filter(regex='^(Crop_Type_|Region_)').columns] = data_dummies[data_dummies.filter(regex='^(Crop_Type_|Region_)')].astype(int)
##data_dummies.iloc[:,14:] = data_dummies.iloc[:,14:].astype(int)
rf_data = data_dummies.iloc[:,2:]

pipelineRF = Pipeline([
    ('scaler',StandardScaler()),
    ('feature selection', SelectKBest(k=12)),
    ('classifier', RandomForestRegressor(n_estimators=500, random_state=1))
])

Y = rf_data['Economic_Impact_Million_USD']
X = rf_data.drop(columns=['Economic_Impact_Million_USD'])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=1)
pipelineRF.fit(X_train,Y_train)
Y_pred = pipelineRF.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print('Model summary for Ranom Forest:')
print('------------------------------------')
print(f'Mean Sqared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-Sqared: {r2}')