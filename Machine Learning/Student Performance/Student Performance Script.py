#Importing packages
import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

pd.set_option('display.max_columns', None)

# Download latest version
path = kagglehub.dataset_download("adilshamim8/student-performance-and-learning-style")

for file in os.listdir(path):
    print(f'The file {file.split('.')[0]} is of type {file.split('.')[1]}.')

#Loading Data
data = pd.read_csv(f'{path}/student_performance_large_dataset.csv')
data = data.rename(columns={'Study_Hours_per_Week':'Weekly Study Hours','Preferred_Learning_Style':'Learning Style','Online_Courses_Completed':'Online Courses',
                            'Participation_in_Discussions':'Discussions Participation','Assignment_Completion_Rate (%)':'Completion Rate','Exam_Score (%)':'Exam Score',
                            'Attendance_Rate (%)':'Attendance','Use_of_Educational_Tech':'Educational Tech', 'Self_Reported_Stress_Level':'Stress level',
                            'Time_Spent_on_Social_Media (hours/week)':'Social Media Time','Sleep_Hours_per_Night':'Sleeping Time','Final_Grade':'Final Grade'})
print(data.head(),'\n')

#Visuals
study_method = data.groupby(['Learning Style','Gender'])['Exam Score'].mean().reset_index()
stress_level = data.groupby(['Stress level','Gender'])['Exam Score'].mean().reset_index()
fig1, (ax1A,ax1B) = plt.subplots(2,1)
sns.barplot(x='Learning Style',y='Exam Score',hue='Gender',data=study_method, ax=ax1A)
sns.barplot(x='Stress level',y='Exam Score',hue='Gender',data=stress_level, ax=ax1B)
ax1A.set_title('Results per Learning Style')
ax1A.set_ylabel('Exam Score (%)')
ax1A.set_xlabel(None)
ax1B.set_title('Results per Stress Level')
ax1B.set_ylabel('Exam Score (%)')
ax1B.set_xlabel(None)

corr = data[['Age','Weekly Study Hours','Online Courses','Completion Rate','Exam Score',
             'Attendance','Social Media Time','Sleeping Time']].corr()
fig3, ax3 = plt.subplots()
sns.heatmap(corr, annot=True, cmap='viridis')
ax3.set_title('Correlation of main variables')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)

plt.show()

#Preprocessing
Sc = StandardScaler()
Le = LabelEncoder()
Oe_stress = OrdinalEncoder(categories=[['Low','Medium','High']])
Oe_grade = OrdinalEncoder(categories=[['C','D','B','A']])

data_enc = data.copy().drop(columns='Student_ID')
data_enc[data_enc.select_dtypes(['number']).columns] = Sc.fit_transform(data_enc[data_enc.select_dtypes(['number']).columns])
data_enc['Gender'] = Le.fit_transform(data_enc['Gender'])
data_enc['Discussions Participation'] = Le.fit_transform(data_enc['Discussions Participation'])
data_enc['Educational Tech'] = Le.fit_transform(data_enc['Educational Tech'])
data_enc['Stress level'] = Oe_stress.fit_transform(data_enc[['Stress level']])
data_enc['Final Grade'] = Oe_grade.fit_transform(data_enc[['Final Grade']])
data_enc = pd.get_dummies(data_enc,columns=['Learning Style'],drop_first=True)

##Regression
#Data split
Y = data_enc['Exam Score']
X = data_enc.drop(columns=['Exam Score', 'Final Grade'])
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=1)

#OLS
ols = LinearRegression()

#Tree
tree_r = DecisionTreeRegressor(max_depth=7)

#Random Forest
rf_r = RandomForestRegressor(n_estimators=250, random_state=1)

#Model performances
models = {}
for model, name in zip([ols, tree_r, rf_r],['Ordinary Least Squares', 'Regression Tree', 'Regression Random Forest']):
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    r2 = r2_score(ytest,ypred)
    mae = mean_absolute_error(ytest,ypred)
    mse = mean_squared_error(ytest,ypred)
    models[name] = [r2, mae, mse]
print(pd.DataFrame(models,index=['R2','MAE','MSE']),'\n')

##Classification
#Data split
Y = data_enc['Final Grade']
X = data_enc.drop(columns=['Exam Score', 'Final Grade'])
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=1)

#Logit
logit = LogisticRegression()

#Tree
tree_c = DecisionTreeClassifier(max_depth=7)

#Random Forest
rf_c = RandomForestClassifier(n_estimators=250, random_state=1)
rf_c.fit(xtrain,ytrain)
importance = rf_c.feature_importances_
print(pd.DataFrame({'Features':xtrain.columns, 'Importance':importance}).sort_values(by='Importance').reset_index(drop=True).head(),'\n')

#Model performances
models = {}
for model, name in zip([logit, tree_c, rf_c],['Logit', 'Classification Tree', 'Classification Random Forest']):
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    acc = accuracy_score(ytest, ypred)
    pre = precision_score(ytest, ypred, average='macro')
    print(f'Classification report for the {name} model:')
    print(classification_report(ytest,ypred),'\n')
    cm = confusion_matrix(ytest,ypred)
    ConfusionMatrixDisplay(cm, display_labels=['C','D','B','A']).plot(cmap='Greens')
    plt.title(f'Cnfusion matrix of the {name} model with an accuracy of {acc}.')
    plt.show()
    models[name] = [acc, pre]
print(pd.DataFrame(models, index=['Accuracy', 'Precission']),'\n')
