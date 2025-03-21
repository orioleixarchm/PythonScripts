#Load Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
pd.set_option('display.max_columns',None)

#Load data
path = r"C:\Users\User\OneDrive\Programació\Python\Kaggle\Marriage Data India".replace("\\","/")
print(os.listdir(path=path),'\n')
data = pd.read_csv(f'{path}/marriage_data_india.csv')
print(data.head(),'\n')

#Data Exploration
for col in data.select_dtypes("object"):
    print(f'Unique Values for Column {col}:')
    print(data[col].unique(),'\n')
data.set_index('ID', inplace=True)

#Transforming binary variables
data_bin = data.copy()
le = LabelEncoder()
for col in data_bin.select_dtypes("object"):
    if len(data_bin[col].unique()) == 2:
        data_bin[f"{col}_enc"] = le.fit_transform(data_bin[col])
print(data_bin.head(),'\n')

#Visualization
def visual(dataset, categorical_I, categorical_II, value, axis, plot=False):
    categories = [categorical_I, categorical_II] if categorical_II else [categorical_I]
    data_grouped = dataset.groupby(categories)[value].mean().round(2).reset_index()

    sns.barplot(data=data_grouped, x=categorical_I, y=value, hue=categorical_II, ax=axis)
    axis.set_title(f"{value} per {categorical_I}")
    if categorical_II:
        axis.legend()
    if plot:
        plt.show()    

fig1, ax1 = plt.subplots(2,2)
visual(dataset=data_bin, categorical_I='Education_Level', categorical_II='Gender', value='Age_at_Marriage', axis=ax1[0,0])
visual(dataset=data_bin, categorical_I='Religion', categorical_II='Gender', value='Urban_Rural_enc', axis=ax1[0,1])
visual(dataset=data_bin, categorical_I='Education_Level', categorical_II='Religion', value='Children_Count', axis=ax1[1,0])
visual(dataset=data_bin, categorical_I='Gender', categorical_II='Religion', value='Years_Since_Marriage', axis=ax1[1,1], plot=True)

#Enoding
Le = LabelEncoder()
Sc = StandardScaler()
Ed_en = OrdinalEncoder(categories=[['School', 'Graduate', 'Postgraduate', 'PhD']])
MS_en = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
PA_en = OrdinalEncoder(categories=[['No', 'Partial', 'Yes']])
IL_en = OrdinalEncoder(categories=[['Low', 'Middle', 'High']])
data_en = data.copy()
data_en[data_en.select_dtypes("int").columns]= Sc.fit_transform(data_en[data_en.select_dtypes("int").columns])
data_en['Education_Level'] = Ed_en.fit_transform(data_en[['Education_Level']])
data_en['Marital_Satisfaction'] = MS_en.fit_transform(data_en[['Marital_Satisfaction']])
data_en['Parental_Approval'] = PA_en.fit_transform(data_en[['Parental_Approval']])
data_en['Income_Level'] = IL_en.fit_transform(data_en[['Income_Level']])
data_en = pd.get_dummies(data=data_en, columns=['Religion','Dowry_Exchanged',], drop_first=True)
for col in data_en.select_dtypes("object"):
    if len(data_bin[col].unique()) == 2:
        data_en[col] = Le.fit_transform(data_en[col])

#Model Split
Y = data_en['Marital_Satisfaction']
X = data_en.drop(columns=['Marital_Satisfaction'])
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=1)

##Classification
#Logistic
logit = LogisticRegression()

#Tree
tree = DecisionTreeClassifier(max_depth=8, random_state=1)

#Random Forest
rf = RandomForestClassifier(n_estimators=250, random_state=1)
rf.fit(xtrain,ytrain)
importance = rf.feature_importances_
print(pd.DataFrame({"Variable":xtrain.columns, "Importance":importance}).sort_values(by='Importance', ascending=False).reset_index(drop=True).head(10),'\n')

#Results
models = {}
for model, name in zip([logit, tree, rf],["Logistic Classification", "Classification Tree", "Random Forest"]):
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    acc = accuracy_score(ytest,ypred)
    pre = precision_score(ytest,ypred, average='weighted')
    cm = confusion_matrix(ytest,ypred)
    ConfusionMatrixDisplay(cm, display_labels=['Low', 'Medium', 'High']).plot(cmap="Greens")
    plt.title(f"Confusion Matrix for the {name} model, with accuracy of {acc}.")
    plt.show()
    print(f"Report of the {name} model:")
    print(classification_report(ytest,ypred),'\n')
    models[name] = [acc, pre]
print(pd.DataFrame(models, index=['Accuracy', 'Precission']))

#Multinomial Logistic Regression
X_const = sm.add_constant(X).astype(int) 
print(Y.head())
print(data['Marital_Satisfaction'].head())
ML_model = sm.MNLogit(Y,X_const).fit()
print(ML_model.summary())