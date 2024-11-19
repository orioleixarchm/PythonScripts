#Loading Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,  precision_score,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

#Loading data
path = 'C:/Users/IU3309/OneDrive - ELIA GROUP/Desktop'
file = 'weather_forecast_data'
type = 'csv'
data = pd.read_csv(f'{path}/{file}.{type}')
print(data.head(),'\n')

#Correlation
cor = data.iloc[:,:-1].corr()
sns.heatmap(cor, annot=True, cmap='Blues')
plt.xticks(rotation=0)
plt.show()

#Encoding
Le = LabelEncoder()
data['Rain'] = Le.fit_transform(data['Rain'])

#Data Split
X = data.drop(columns="Rain")
Y = data["Rain"]
xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.2, random_state=1)

#Logistic Regression
logit = LogisticRegression()

#Decision Tree
tree = DecisionTreeClassifier(max_depth=8)

#Random Forest
rfc = RandomForestClassifier(n_estimators=250, random_state=1)
rfc.fit(xtrain,ytrain)
importance = rfc.feature_importances_
print(pd.DataFrame({'Importance': importance}, index=xtest.columns).sort_values(by='Importance', ascending=False).head(),'\n')

#Models
models = {}
for model, name in zip([logit, tree, rfc], ["Logistic Classification", "Decision Tree", "Random Forest"]):
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    acc = accuracy_score(ytest, ypred)
    prescore = precision_score(ytest, ypred)
    models[name] = [acc, prescore]
    print(f'Classification report for the {name} model:')
    print(classification_report(ytest, ypred), '\n')
    cm = confusion_matrix(ytest, ypred)
    ConfusionMatrixDisplay(cm, display_labels=["No Rain", "Rain"]).plot(cmap="Blues")
    plt.title(f'Confusion Matrix for the {name} model:')
    plt.show()

print(pd.DataFrame(models, index=["Accuracy", "Precision"]),'\n')