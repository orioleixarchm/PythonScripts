#Importing packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.max_columns', None)

#Variables
variables = {
    'SalePrice': 'The property sale price in dollars',
    'MSSubClass': 'The building class',
    'MSZoning': 'The general zoning classification',
    'LotFrontage': 'Linear feet of street connected to property',
    'LotArea': 'Lot size in square feet',
    'Street': 'Type of road access',
    'Alley': 'Type of alley access',
    'LotShape': 'General shape of property',
    'LandContour': 'Flatness of the property',
    'Utilities': 'Type of utilities available',
    'LotConfig': 'Lot configuration',
    'LandSlope': 'Slope of property',
    'Neighborhood': 'Physical locations within Ames city limits',
    'Condition1': 'Proximity to main road or railroad',
    'Condition2': 'Proximity to main road or railroad (if a second is present)',
    'BldgType': 'Type of dwelling',
    'HouseStyl': 'Style of dwelling',
    'OverallQual': 'Overall material and finish quality',
    'OverallCond': 'Overall condition rating',
    'YearBuilt': 'Original construction date',
    'YearRemodAdd': 'Remodel date',
    'RoofStyle': 'Type of roof',
    'RoofMatl': 'Roof material',
    'Exterior1st': 'Exterior covering on house',
    'Exterior2nd': 'Exterior covering on house (if more than one material)',
    'MasVnrType': 'Masonry veneer type',
    'MasVnrArea': 'Masonry veneer area in square feet',
    'ExterQual': 'Exterior material quality',
    'ExterCond': 'Present condition of the material on the exterior',
    'Foundation': 'Type of foundation',
    'BsmtQual': 'Height of the basement',
    'BsmtCond': 'General condition of the basement',
    'BsmtExposure': 'Walkout or garden level basement walls',
    'BsmtFinType1': 'Quality of basement finished area',
    'BsmtFinSF1': 'Type 1 finished square feet',
    'BsmtFinType2': 'Quality of second finished area (if present)',
    'BsmtFinSF2': 'Type 2 finished square feet',
    'BsmtUnfSF': 'Unfinished square feet of basement area',
    'TotalBsmtSF': 'Total square feet of basement area',
    'Heating': 'Type of heating',
    'HeatingQC': 'Heating quality and condition',
    'CentralAir': 'Central air conditioning',
    'Electrical': 'Electrical system',
    '1stFlrSF': 'First Floor square feet',
    '2ndFlrSF': 'Second floor square feet',
    'LowQualFinSF': 'Low quality finished square feet (all floors)',
    'GrLivArea': 'Above grade (ground) living area square feet',
    'BsmtFullBath': 'Basement full bathrooms',
    'BsmtHalfBath': 'Basement half bathrooms',
    'FullBath': 'Full bathrooms above grade',
    'HalfBath': 'Half baths above grade',
    'Bedroom': 'Number of bedrooms above basement level',
    'Kitchen': 'Number of kitchens',
    'KitchenQual': 'Kitchen quality',
    'TotRmsAbvGrd': 'Total rooms above grade (does not include bathrooms)',
    'Functional': 'Home functionality rating',
    'Fireplaces': 'Number of fireplaces',
    'FireplaceQu': 'Fireplace quality',
    'GarageType': 'Garage location',
    'GarageYrBlt': 'Year garage was built',
    'GarageFinish': 'Interior finish of the garage',
    'GarageCars': 'Size of garage in car capacity',
    'GarageArea': 'Size of garage in square feet',
    'GarageQual': 'Garage quality',
    'GarageCond': 'Garage condition',
    'PavedDrive': 'Paved driveway',
    'WoodDeckSF': 'Wood deck area in square feet',
    'OpenPorchSF': 'Open porch area in square feet',
    'EnclosedPorch': 'Enclosed porch area in square feet',
    '3SsnPorch': 'Three season porch area in square feet',
    'ScreenPorch': 'Screen porch area in square feet',
    'PoolArea': 'Pool area in square feet',
    'PoolQC': 'Pool quality',
    'Fence': 'Fence quality',
    'MiscFeature': 'Miscellaneous feature not covered in other categories',
    'MiscVal': '$Value of miscellaneous feature',
    'MoSold': 'Month Sold',
    'YrSold': 'Year Sold',
    'SaleType': 'Type of sale',
    'SaleCondition': 'Condition of sale'
}

#Loading and splitting data
path = r'C:\Users\User\OneDrive\Programació\Python\Kaggle\House_pricing'.replace('\\','/')
data = pd.read_csv(path + '/House-Price-Prediction-clean.csv')
print(data.head(),'\n')
print(data.info(),'\n')
X = data.drop(columns='SalePrice')
X_lasso_raw = PolynomialFeatures(degree=2, include_bias=False)
X_lasso = X_lasso_raw.fit_transform(X)
Y = data['SalePrice']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1)
X_train_lasso, X_test_lasso, _ , _ = train_test_split(X_lasso,Y, test_size=0.2, random_state=1)

#Preprocessing
lasso_pipe = Pipeline(steps=[
    ("Scaler", StandardScaler()),
    ("Imputer", SimpleImputer(strategy='median')),
    ("Lasso", Lasso(alpha=0.1, max_iter=100000, random_state=1))
])

tree_pipe = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='median')),
    ('Tree', DecisionTreeRegressor(random_state=1, max_depth=15))
])

forest_pipe = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='median')), 
    ('Forest', RandomForestRegressor(random_state=1, n_estimators=300, max_depth=15))
])

#Tunning models
cross_val = KFold(n_splits=5, shuffle=True, random_state=1)
lasso_arg = {
    'Lasso__alpha':[500,525,550,575,600,625]
    }
grid_lasso = GridSearchCV(lasso_pipe, lasso_arg, cv=cross_val,scoring='neg_root_mean_squared_error',refit=True, n_jobs=-1)
grid_lasso.fit(X_train_lasso, Y_train)
print(f'Best alpha is: {grid_lasso.best_params_}, with a score (CVRMSE) of {-grid_lasso.best_score_}.','\n')
Lasso = grid_lasso.best_estimator_

tree_pipe.fit(X_train, Y_train)

rf_arg = {
    'Forest__max_depth':[5,10,15,20,25,30],
    'Forest__n_estimators':[200,400,600]
    }
grid_forest = GridSearchCV(forest_pipe, rf_arg, cv=cross_val,scoring='neg_root_mean_squared_error',refit=True, n_jobs=-1)
grid_forest.fit(X_train, Y_train)
print(f'Best alpha are: {grid_forest.best_params_}, with a score (CVRMSE) of {-grid_forest.best_score_}.','\n')
random_forest = grid_forest.best_estimator_

#Testing, scores and metrics
models = {}
def evaluation(nombre, model, xtest, ytest):
    y_pred = model.predict(xtest)
    rmse = mean_squared_error(ytest, y_pred=y_pred)
    r2 = r2_score(ytest, y_pred=y_pred)
    print(f'{nombre} model performance: RMSE = {rmse}, R2 = {r2}.','\n')
    models[nombre] = (round(rmse,3), round(r2,3))

evaluation('Lasso', Lasso, X_test_lasso, Y_test)
evaluation('Decision Tree', tree_pipe, X_test, Y_test)
evaluation('Random Forest', random_forest, X_test, Y_test)
print(pd.DataFrame(models, index=['RMSE','R2']),'\n')

#Feature importance
lasso_features_importance = pd.DataFrame({
    'Variable': X_lasso_raw.get_feature_names_out(X.columns),
    'Coefficient': Lasso.named_steps['Lasso'].coef_,
    'AbsCoef': np.abs(Lasso.named_steps['Lasso'].coef_)
}).sort_values(by='AbsCoef', ascending=False).drop(columns='AbsCoef').reset_index(drop=True)

tree_features_importance = pd.DataFrame({
    'Variable':X.columns,
    'Importance': tree_pipe.named_steps['Tree'].feature_importances_
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

forest_features_importance = pd.DataFrame({
    'Variable': X.columns,
    'Importance': random_forest.named_steps['Forest'].feature_importances_
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

print('##### LASSO #####')
print(lasso_features_importance.head(10),'\n')
print('##### Regression Tree #####')
print(tree_features_importance.head(10),'\n')
print('##### Random Forest #####')
print(forest_features_importance.head(10),'\n')