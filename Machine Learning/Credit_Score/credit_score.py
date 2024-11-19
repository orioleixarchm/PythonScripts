#Loading packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, accuracy_score,
                             classification_report, roc_auc_score, precision_score)

pd.set_option('display.width', None)

#Loading data
path = 'C:/Users/oriol/OneDrive/Programació/Python/Kaggle/Credit_Score'
file = 'credit_score_cleaned_train'
type = 'csv'
data = pd.read_csv(f'{path}/{file}.{type}')
print(data.head(),'\n')

#Data exploration and preparation
print(data.isnull().sum(),'\n')
print(data.head()['type_of_loan'],'\n')
data['type_of_loan'] = data['type_of_loan'].str.replace(r"[']", "", regex=True).str.replace(" ", "_").str.replace(", ", "_/_")
print(data.head()['type_of_loan'],'\n')
data['delay_from_due_date'] = abs(data['delay_from_due_date'])
print(data['credit_score'].unique(),'\n')

#Visualization
data_score = data.select_dtypes(include=['number']).groupby('credit_score').mean()
data_score.index = data_score.index.map({0:'Low', 1:'Medium', 2:'High'})
print(data_score.head(),'\n')
columns = ['age', 'annual_income', 'num_bank_accounts', 'interest_rate', 'num_of_delayed_payment', 'delay_from_due_date']
for item, name in zip(columns, ['Age', 'Annual Income', 'Active bank Accounts', 'Interest Rate',
                                'Nº Delayed Payments', 'Nº Days Delayed']):
    plt.bar(data_score.index, data_score[item])
    plt.ylabel(f'{name}')
    plt.yticks(None)
    plt.title(f'{name} per Credit Score Level')
    plt.show()

corr = data[columns].corr()
sns.heatmap(corr, annot=True)
plt.title('Correlation Matrix')
plt.xticks(rotation=0)
plt.show()

#Encoding
data_ml = data.drop(columns=['id', 'customer_id', 'month', 'name', 'ssn', 'type_of_loan'])
data_ml['credit_score'] = data_ml['credit_score'].map({0:'Bad', 1:'Standard', 2:'Good'})
Ss = StandardScaler()
Le = LabelEncoder()
Oe = OrdinalEncoder(categories=[['Bad', 'Standard', 'Good']])

columns_to_scale = data_ml.select_dtypes(include=['number'])
columns_to_Lencode = 'payment_of_min_amount'
columns_to_Oencode = ['credit_mix', 'credit_score']
columns_to_onehot_enc = ['payment_behaviour', 'occupation']

data_ml[columns_to_scale.columns] = Ss.fit_transform(data_ml[columns_to_scale.columns])
data_ml[columns_to_Lencode] = Le.fit_transform(data_ml[columns_to_Lencode])
for column in columns_to_Oencode:
    data_ml[column] = Oe.fit_transform(data_ml[[column]])
data_ml = pd.get_dummies(data_ml, columns=columns_to_onehot_enc, drop_first=True)
print(data_ml.head(),'\n')

#Spliting data
Y = data_ml['credit_score']
X = data_ml.drop(columns='credit_score')
xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.2, random_state=1)

#Logistic Classification
logit = LogisticRegression(multi_class='ovr')

#Decision Tree
tree = DecisionTreeClassifier(max_depth=5)

#RandomForest
rforest = RandomForestClassifier(n_estimators=200, random_state=1)
rforest.fit(xtrain,ytrain)
importance = rforest.feature_importances_
print(pd.DataFrame({'Importance':importance}, index=xtest.columns).sort_values(by='Importance', ascending=False),'\n')


#Results
models = {}
for model, name in zip([logit, tree, rforest], ['Logistic Class.', 'Decision Tree', 'Random Forest']):
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    acc = accuracy_score(ytest, ypred)
    print(f'Report for the {name} model:')
    print(classification_report(ytest, ypred))
    precision = precision_score(ytest, ypred, average='macro')
    cm = confusion_matrix(ytest, ypred)
    ConfusionMatrixDisplay(cm, display_labels=['Bad', 'Standard', 'Good']).plot(cmap='Greens')
    plt.title(f'Confusion Matrix of the {name} model with Accuracy of: {acc}.')
    plt.show()
    models[name] = [acc, precision]

print(pd.DataFrame(models, index=['Accuracy', 'Precision']),'\n')
