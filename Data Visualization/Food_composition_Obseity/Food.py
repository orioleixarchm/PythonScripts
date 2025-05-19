#Importing packges
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import os


#Loading data
path = r'C:\Users\User\OneDrive\Programació\Python\Kaggle\Food'.replace('\\','/')
print(os.listdir(path),'\n')
file_names = ['Fat_Supply_Quantity_Data', 'Food_Supply_kcal_Data', 'Food_Supply_Quantity_kg_Data', 'Protein_Supply_Quantity_Data', 'Supply_Food_Data_Descriptions']
files = {}
for doc in file_names:
    files[doc] = pd.read_csv(f'{path}/{doc}.csv')
    print(f'Dataset: {doc}:')
    print(files[doc].head(),'\n')
    print(files[doc].info())

Fats = files['Fat_Supply_Quantity_Data']
Quantity = files['Food_Supply_Quantity_kg_Data']
Protein = files['Protein_Supply_Quantity_Data']
Calories = files['Food_Supply_kcal_Data']
Calories['Diversification'] = 1-(Calories.iloc[:,1:-8].div(100)**2).sum(axis=1)
Calories['Undernourished'] = Calories['Undernourished'].apply(lambda x: np.random.uniform(0, 2.49) if x == '<2.5' else float(x))
Calories['Total Obese'] = (Calories['Obesity'].div(100)*Calories['Population']).div(1000000)
Calories['Total Undernourished'] = Calories['Undernourished'].div(100)*Calories['Population']
Calories['Population'] = Calories['Population'].div(1000000)
Veg_cols = ['Cereals - Excluding Beer', 'Eggs', 'Fruits - Excluding Wine',
            'Milk - Excluding Butter', 'Miscellaneous', 'Oilcrops','Offals', 
	    'Pulses', 'Spices', 'Stimulants','Starchy Roots', 'Sugar & Sweeteners',
            'Sugar Crops', 'Treenuts', 'Vegetable Oils','Vegetables', 'Vegetal Products']
Calories['Prop Anim'] = 100-(Calories[Veg_cols].sum(axis=1))
Calories_dev = Calories.loc[Calories['Undernourished']<=Calories['Undernourished'].median(),Veg_cols + ['Diversification','Obesity','Deaths']]

fig_I, axis_I = plt.subplots(2,2)
fig_I.tight_layout(pad=2)
fig_I.suptitle('Vegeterian-like & carnivore-like diets vs Obesity')
sns.regplot(data=Calories, x='Vegetables', y='Obesity', ax=axis_I[0,0], color='darkgreen')
sns.scatterplot(data=Calories, x='Vegetables', y='Obesity', ax=axis_I[0,0], color='darkgreen', edgecolor='black', linewidth=1)
axis_I[0,0].set_title('Obesity Rate vs Rate of Vegetables consumption')
axis_I[0,0].set_xlabel('Percentage (%) of calories comming from vegetables')
axis_I[0,0].set_ylabel('Obesity Rate (%)')

sns.regplot(data=Calories, x='Vegetal Products', y='Obesity', ax=axis_I[1,0], color='green')
sns.scatterplot(data=Calories, x='Vegetal Products', y='Obesity', ax=axis_I[1,0], color='green', edgecolor='black', linewidth=1)
axis_I[1,0].set_title('Obesity Rate vs Rate of Vegetal Products consumption')
axis_I[1,0].set_xlabel('Percentage (%) of calories comming from vegetal products')
axis_I[1,0].set_ylabel('Obesity Rate (%)')

sns.regplot(data=Calories, x='Meat', y='Obesity', ax=axis_I[0,1], color='darkred')
sns.scatterplot(data=Calories, x='Meat', y='Obesity', ax=axis_I[0,1], color='darkred', edgecolor='black', linewidth=1)
axis_I[0,1].set_title('Obesity Rate vs Rate of Meat consumption')
axis_I[0,1].set_xlabel('Percentage (%) of calories comming from meat')
axis_I[0,1].set_ylabel('Obesity Rate (%)')

sns.regplot(data=Calories, x='Animal Products', y='Obesity', ax=axis_I[1,1], color='red')
sns.scatterplot(data=Calories, x='Animal Products', y='Obesity', ax=axis_I[1,1], color='red', edgecolor='black', linewidth=1)
axis_I[1,1].set_title('Obesity Rate vs Rate of Animal Products consumption')
axis_I[1,1].set_xlabel('Percentage (%) of calories comming from animal products')
axis_I[1,1].set_ylabel('Obesity Rate (%)')


fig_II, axis_II = plt.subplots(2,2)
fig_II.tight_layout(pad=2)
fig_II.suptitle('Impact of Diversified diet')
sns.regplot(data=Calories_dev, x='Diversification',y='Obesity', ax=axis_II[0,0], color='lightblue')
sns.scatterplot(data=Calories_dev, x='Diversification',y='Obesity', ax=axis_II[0,0], color='lightblue', edgecolor='black', linewidth=1)
axis_II[0,0].set_title('Diet Diversification Rate vs Obesity Rate')
axis_II[0,0].set_xlabel('Diversification Index')
axis_II[0,0].set_ylabel('Obesity Rate (%)')

sns.regplot(data=Calories_dev, x='Diversification',y='Deaths', ax=axis_II[0,1], color='blue')
sns.scatterplot(data=Calories_dev, x='Diversification',y='Deaths', ax=axis_II[0,1], color='blue', edgecolor='black', linewidth=1)
axis_II[0,1].set_title('Diet Diversification Rate vs Death Rate')
axis_II[0,1].set_xlabel('Diversification Index')
axis_II[0,1].set_ylabel('Death Rate (%)')
axis_II[0,1].set_ylim(-0.02, Calories['Deaths'].max()*1.1)

sns.regplot(data=Calories, x='Obesity',y='Deaths', ax=axis_II[1,0], color='darkblue')
sns.scatterplot(data=Calories, x='Obesity',y='Deaths', ax=axis_II[1,0], color='darkblue', edgecolor='black', linewidth=1)
axis_II[1,0].set_title('Obesity Rate vs Death Rate')
axis_II[1,0].set_xlabel('Obesity Rate (%)')
axis_II[1,0].set_ylabel('Death Rate (%)')

sns.regplot(data=Calories, x='Prop Anim',y='Obesity', ax=axis_II[1,1], color='SteelBlue')
sns.scatterplot(data=Calories, x='Prop Anim',y='Obesity', ax=axis_II[1,1], color='SteelBlue', edgecolor='black', linewidth=1)
axis_II[1,1].set_title('Animal Product Consumption vs Obesity Rate')
axis_II[1,1].set_xlabel('Animal Origin Consumption')
axis_II[1,1].set_ylabel('Obesity Rate (%)')


fig_III, axes_III = plt.subplots(1,2)
fig_III.tight_layout(pad=2)
fig_III.suptitle('Most Obese Countries')
sns.barplot(data=Calories.loc[:,['Country','Obesity']].sort_values(by='Obesity',ascending=False).head(), x='Country',y='Obesity', ax=axes_III[0], edgecolor='black', linewidth=1, color='SteelBlue')
axes_III[0].set_title('Countries with higher Obesity Rates')
axes_III[0].set_ylabel('Obesity Rate (%)')
axes_III[0].set_xlabel(None)

sns.barplot(data=Calories.loc[:,['Country','Total Obese']].sort_values(by='Total Obese',ascending=False).head(), x='Country',y='Total Obese', ax=axes_III[1], edgecolor='black', linewidth=1, color='darkblue')
axes_III[1].set_title('Countries with more Obese people')
axes_III[1].set_ylabel('Nº Obese people (in Millions)')
axes_III[1].set_xlabel(None)

Variables = ['Animal Products', 'Animal fats', 'Eggs',
       'Fish, Seafood', 'Fruits - Excluding Wine', 'Meat', 
       'Oilcrops', 'Pulses', 'Spices', 'Starchy Roots', 'Sugar Crops', 'Sugar & Sweeteners', 
       'Treenuts', 'Vegetal Products', 'Vegetable Oils', 'Vegetables']

fig_IV, ax = plt.subplots()
fig_IV.tight_layout(pad=4)
sns.heatmap(Calories[Variables].corr(), annot=True)
plt.xticks( rotation=55)
plt.title('Correlation Among main variables')
plt.show()


#OLS with Centered Log-Ratio (CLR) transformation
Calories_var = ['Obesity', 'Alcoholic Beverages', 'Animal Products', 'Animal fats',
       'Aquatic Products, Other', 'Cereals - Excluding Beer', 'Eggs',
       'Fish, Seafood', 'Fruits - Excluding Wine', 'Meat',
       'Milk - Excluding Butter', 'Miscellaneous', 'Offals', 'Oilcrops',
       'Pulses', 'Spices', 'Starchy Roots', 'Stimulants', 'Sugar Crops',
       'Sugar & Sweeteners', 'Treenuts', 'Vegetal Products', 'Vegetable Oils',
       'Vegetables']
Calories_ols = Calories[Calories_var].replace(0,1e-6).dropna()

Y = Calories_ols['Obesity']
X = Calories_ols.drop(columns=['Obesity'])
geomean = X.apply(lambda x: np.exp(np.mean(np.log(x))), axis=1)
X_clr = np.log(X.div(geomean, axis=0))
model = sm.OLS(Y, X_clr).fit()
print(model.summary(),'\n')


