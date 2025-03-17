#Loading Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, LassoCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score, r2_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

#Loading and exploring data
path = 'C:/Users/oriol/OneDrive/Programació/Python/Kaggle/Grape Quality'
file = 'GRAPE_QUALITY'
filetype = 'csv'
data = pd.read_csv(f'{path}/{file}.{filetype}')
print(data.head(),'\n')
print(data.info(),'\n')
print('Unique Education Values:', data['variety'].unique(),'\n')
print('Unique Job Title Values:', data['region'].unique(),'\n')

# Aggregates and times
components = data[['variety', 'region', 'quality_score', 'sugar_content_brix', 'acidity_ph', 'cluster_weight_g', 'berry_size_mm', 'sun_exposure_hours', 'soil_moisture_percent', 'rainfall_mm']]
components_region = components.drop(columns="variety").groupby('region').mean()
components_variety = components.drop(columns="region").groupby('variety').mean()
timeline = data[['quality_score','harvest_date']]
timeline['harvest_date'] = pd.to_datetime(timeline['harvest_date'])
timeline = timeline.groupby('harvest_date').mean()
timeline = timeline.resample('D').asfreq()
correlation_m = components.drop(columns=['variety', 'region']).corr()
timeline.index = timeline.index.strftime('%m-%d')
print(components_region,'\n',components_variety,'\n')

#Visuals
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data['quality_score'], data['sugar_content_brix'], data['acidity_ph'], cmap='viridis')
ax.set_xlabel('Quality Score')
ax.set_ylabel('Sugar Content')
ax.set_zlabel('Acidity')
plt.show()

plt.figure()
sns.heatmap(correlation_m, annot=True)
plt.title('Correlation Matrix')
plt.show()

timeline.plot()
plt.xlabel('Day of harvest')
plt.ylabel('Quality Score')
plt.title('Quality on Day of the harvest')
plt.show()

#Encoding
Sc = StandardScaler()
Oe = OrdinalEncoder(categories=[['Low','Medium','High','Premium']])
data_enc = data.copy()
data_enc[['quality_score', 'sugar_content_brix', 'acidity_ph', 'cluster_weight_g', 'berry_size_mm', 'sun_exposure_hours', 'soil_moisture_percent', 'rainfall_mm']] = Sc.fit_transform(data_enc[['quality_score', 'sugar_content_brix', 'acidity_ph', 'cluster_weight_g', 'berry_size_mm', 'sun_exposure_hours', 'soil_moisture_percent', 'rainfall_mm']])
data_enc['quality_category'] = Oe.fit_transform(data_enc[['quality_category']])
data_enc = pd.get_dummies(data_enc, columns=['variety', 'region'], drop_first=True)
print(data_enc.head(),'\n')

##Regression
#Data Split for Regression
Y = data_enc['quality_score']
X = data_enc.drop(columns=['quality_score', 'quality_category', 'harvest_date'])
xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.2, random_state=1)

#Ordinary Least Sqares
ols = LinearRegression()

#Lasso Cross Validated
lasso = LassoCV(alphas=np.logspace(-4,4,100), cv=5, random_state=1)

#Decision Tree
treer = DecisionTreeRegressor(max_depth=7)

#Random Forest
rfr = RandomForestRegressor(n_estimators=250, random_state=1)
rfr.fit(xtest,ytest)
y_pred = rfr.predict(xtest)
importances = rfr.feature_importances_
print(pd.DataFrame({'Variables':xtest.columns,
                    'Importance':importances}).sort_values(by='Importance', ascending=False).head(),'\n')

modelsr = {}
for model, nmodel in zip([ols, lasso, treer, rfr],['OLS', 'Lasso', 'Regression Tree', 'Regression Random Forest']):
    model.fit(xtrain,ytrain)
    y_pred = model.predict(xtest)
    mse = mean_squared_error(ytest, y_pred)
    r2 = r2_score(ytest, y_pred)
    mae = mean_absolute_error(ytest, y_pred)
    mape = mean_absolute_percentage_error(ytest, y_pred)
    evs = explained_variance_score(ytest, y_pred)
    print(f'{nmodel} presents a MSE of {round(mse,3)} and a R2 of {round(r2,3)}.','\n')
    modelsr[nmodel] = [mse, mae, mape, r2, evs]

print(pd.DataFrame(modelsr, index=['MSE','MAE','MAPE','R2','EVS']).head(),'\n')

##Classification
#Data Split for Classification
Y = data_enc['quality_category']
X = data_enc.drop(columns=['quality_score', 'quality_category', 'harvest_date'])
xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.3, random_state=1)

#Logistic
logit = LogisticRegression(max_iter=250)

#Decision Tree
treec = DecisionTreeClassifier(max_depth=7)

#Random Forest
rfc = RandomForestClassifier(n_estimators=250, random_state=1)

modelsc = {}
for model, nmodel in zip([logit,treec,rfc], ['Logit', 'Classification Tree', 'Random Forest']):
    model.fit(xtrain,ytrain)
    model_y = model.predict(xtest)
    model_acscore = accuracy_score(ytest, model_y)
    class_report_model = classification_report(ytest, model_y)
    cm = confusion_matrix(ytest, model_y)
    ConfusionMatrixDisplay(cm,display_labels=['Low', 'Medium', 'High','Premium']).plot(cmap='Blues')
    plt.title(f'Confusion Matrix of the {nmodel} model with ACC of {round(model_acscore,2)}')
    plt.show()
    modelsc[nmodel] = model_acscore  

print(pd.DataFrame(modelsc, index=['Accuracy Score']))