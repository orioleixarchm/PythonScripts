#Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, accuracy_score, precision_score, 
                            confusion_matrix, ConfusionMatrixDisplay, classification_report)

pd.set_option('display.max_columns', None)

#Loading data
path = "C:/Users/oriol/OneDrive/Programació/Python/Kaggle/Apple quality"
file = "apple_quality"
type = "csv"
data = pd.read_csv(f'{path}/{file}.{type}')
print(data.head(), '\n')

#Data exploration
print(data.dtypes, '\n')
print(f'Different quality scores: {data['Quality'].unique()}', '\n')
print(f'Variables: {data.shape[1]}, and rows {data.shape[0]}', '\n')
print('Number of missings:', '\n', data.isna().sum(), '\n')
data.drop(columns=['A_id'], inplace=True)
data.dropna(inplace=True)
data['Acidity'] = data['Acidity'].astype(float)

#Visualization
sns.countplot(data, x='Quality')
plt.title('Good and Bad quality totals')
plt.ylabel(None)
plt.show()

sns.pairplot(data=data)
plt.show()

corr = data.drop(columns='Quality').corr()
sns.heatmap(corr, cmap='Blues', annot=True)
plt.title('Correlation between variables')
plt.xticks(rotation=45)
plt.show()

#Encoding
Le = LabelEncoder()
Sc = StandardScaler()
data['Quality'] = Le.fit_transform(data['Quality'])
data.iloc[:,:-1] = Sc.fit_transform(data.iloc[:,:-1])

#Data Split
Y = data['Quality']
X = data.drop(columns='Quality')
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=1)

#Logistic
logit = LogisticRegression()

#Decision Tree
tree = DecisionTreeClassifier(max_depth=8)

#Random forest
rf = RandomForestClassifier(n_estimators=350, random_state=1)
rf.fit(xtrain, ytrain)
importances = rf.feature_importances_
print(pd.DataFrame({'Importance': importances}, index=xtrain.columns).sort_values(by='Importance',ascending=False),'\n')

#Fitting
models = {}
for model, name in zip([logit, tree, rf], ['Logistic', 'Classification Tree', 'Random Forest']):
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    acc = accuracy_score(ytest,ypred)
    prec = precision_score(ytest,ypred)
    cm = confusion_matrix(ytest,ypred)
    ConfusionMatrixDisplay(cm, display_labels=['Good','Bad']).plot(cmap='Blues')
    plt.title(f'Confusion Matrix for the {name} model, with an accuracy of: {acc}')
    plt.show()
    print(f'Repport of the {name} model:')
    print(classification_report(ytest, ypred), '\n')
    models[name] = [acc, prec]

print(pd.DataFrame(models, index=["Accuracy","Precision"]))
