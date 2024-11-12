#Loading packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from scipy import stats
from scipy.stats import shapiro, levene, kruskal


#Loading data
path = 'C:/Users/oriol/OneDrive/Programació/Python/Kaggle/Remote Work and Mental Health'
file = 'Impact_of_Remote_Work_on_Mental_Health.csv' 
data = pd.read_csv(f'{path}/{file}')
pd.set_option('display.max_columns', None)

#Exploring data
print(data.head())
print(data.info())

for column in ['Physical_Activity','Sleep_Quality','Work_Life_Balance_Rating','Company_Support_for_Remote_Work','Mental_Health_Condition','Satisfaction_with_Remote_Work']:
    print(f'{column}:')
    print(data[column].unique())
    print('-------------------------------------------\n\n')

#Cleaning and encoding
data.loc[data['Physical_Activity'].isna(),'Physical_Activity'] = 'None'
data.loc[data['Mental_Health_Condition'].isna(),'Mental_Health_Condition'] = 'None'
ordinal_encoder_P = OrdinalEncoder(categories=[['None', 'Weekly', 'Daily']])
ordinal_encoder_S = OrdinalEncoder(categories=[['Poor', 'Average', 'Good']])
ordinal_encoder_St = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
ordinal_encoder_Mh = OrdinalEncoder(categories=[['None', 'Anxiety', 'Burnout', 'Depression']])
data['Physical_Activity_Enc'] = ordinal_encoder_P.fit_transform(data[['Physical_Activity']])
data['Sleep_Quality_Enc'] = ordinal_encoder_S.fit_transform(data[['Sleep_Quality']])
data['Stress_Level_Enc'] = ordinal_encoder_St.fit_transform(data[['Stress_Level']])
data['Mental_Health_Condition_Enc'] = ordinal_encoder_Mh.fit_transform(data[['Mental_Health_Condition']])
Productivity_Change_map = {'Decrease':-1,'No Change':0,'Increase':1}
Satisfaction_with_Remote_Work_map = {'Unsatisfied':-1,'Neutral':0,'Satisfied':1}
data['Productivity_Change_Enc'] = data['Productivity_Change'].map(Productivity_Change_map)
data['Satisfaction_with_Remote_Work_Enc'] = data['Satisfaction_with_Remote_Work'].map(Satisfaction_with_Remote_Work_map)

#Subseting
subset = data[['Employee_ID','Age','Gender','Job_Role','Industry','Years_of_Experience','Stress_Level',
               'Hours_Worked_Per_Week','Productivity_Change','Physical_Activity','Sleep_Quality','Work_Location',
               'Stress_Level_Enc','Productivity_Change_Enc','Physical_Activity_Enc','Sleep_Quality_Enc','Region',
               'Mental_Health_Condition_Enc', 'Satisfaction_with_Remote_Work_Enc','Company_Support_for_Remote_Work',
               'Work_Life_Balance_Rating']]

#Agreggate Anlaysis
prod_per_work_location = subset.groupby('Work_Location')[['Productivity_Change_Enc','Physical_Activity_Enc','Sleep_Quality_Enc','Stress_Level_Enc','Hours_Worked_Per_Week','Employee_ID']].agg({
                                                            'Productivity_Change_Enc':'mean',
                                                            'Physical_Activity_Enc':'mean',
                                                            'Sleep_Quality_Enc':'mean',
                                                            'Stress_Level_Enc':'mean',
                                                            'Hours_Worked_Per_Week':'mean',
                                                            'Employee_ID':'count'
})
print('Features per Work Location')
print(prod_per_work_location)
print('--------------------------------------------------------------------------\n\n\n')

mental_issues = pd.DataFrame()
mental_issues['Work_Location'] = subset['Work_Location']
mental_issues['Any'] = np.where(subset['Mental_Health_Condition_Enc'] == 0, 0, 1)
mental_issues_location = round(mental_issues.groupby('Work_Location').mean()*100)
print('Mental health issues per Work Location')
print(mental_issues_location)
print('--------------------------------------------------------------------------\n\n\n')

#Testing
##Conditions for ANOVA
for column in ['Productivity_Change_Enc','Physical_Activity_Enc','Sleep_Quality_Enc','Stress_Level_Enc','Hours_Worked_Per_Week']:
    grouped = subset.groupby('Work_Location')[column].transform('mean')
    residuals = subset[column] - grouped
    stat, p_value = shapiro(residuals)
    print(f'{column}:')
    print(f"Normality errors with Shapiro-Wilk test p-value:", p_value) #If p_value > 0.05, the residuals are approximately normal.
    group_values = [group[column].values for name, group in subset.groupby('Work_Location')]
    stat, p_value = levene(*group_values)
    print(f"Homoskedasticity with Levene's test p-value:", p_value) #If p_value > 0.05, you can assume homogeneity of variances.
    f_stat_prod, p_val_prod = stats.f_oneway(*[group[column].values for name, group in subset.groupby('Work_Location')])
    print(f"{column} ANOVA p-value:", p_val_prod)
    stat, p_value = kruskal(*[group[column].values for name, group in subset.groupby('Work_Location')])
    print("Kruskal-Wallis test p-value:", p_value)
    print('--------------------------------------------------------------------------\n')

#Visualization
visual1 = subset.groupby('Years_of_Experience')['Hours_Worked_Per_Week'].mean().reset_index().sort_values(by='Years_of_Experience')
fig, axes = plt.subplots(2,1)
axes[0].bar(visual1['Years_of_Experience'], visual1['Hours_Worked_Per_Week'])
sns.regplot(data=visual1, x='Years_of_Experience', y='Hours_Worked_Per_Week', ax=axes[1])
axes[0].set_title('Avg Hours Worked per Years of experience')
axes[1].set_xlabel('Years Worked')
axes[1].set_ylabel('Avg Hours worked')
plt.show()

visual2 = round(subset.groupby('Work_Location')[['Physical_Activity_Enc','Sleep_Quality_Enc','Stress_Level_Enc']].mean()*100)
fig1, axes1 = plt.subplots(1,3)
columns = ['Physical_Activity_Enc','Sleep_Quality_Enc','Stress_Level_Enc']
yvals = ['Avg Physical Activity Enc','Sleep Quality Enc','Stress Level Enc']
colors = ['green','green','red']
for index, (column, yval, color) in enumerate(zip(columns, yvals, colors)):
    axes1[index].bar(visual2.index, visual2[column], color=color)
    axes1[index].set_title(yval)
    axes1[index].set_xlabel('Work Location')
plt.show()

visual3a = round(subset.groupby('Work_Location')['Mental_Health_Condition_Enc'].mean().mul(100).reset_index())
visual3b = mental_issues_location
fig2, axes2 = plt.subplots(1,2)
axes2[0].barh(visual3a['Work_Location'], visual3a['Mental_Health_Condition_Enc'], color='red')
axes2[0].axvline(x=max(visual3a['Mental_Health_Condition_Enc']), color='black', linestyle='--', linewidth=1)
axes2[0].set_xlabel('Severity of Mental Health')
axes2[0].set_ylabel('Work Location')
axes2[0].set_title('Mental health Severity')
axes2[1].barh(visual3b.index, visual3b['Any'], color='red')
axes2[1].axvline(x=max(visual3b['Any']), color='black', linestyle='--', linewidth=1)
axes2[1].set_xlabel('% of Mental issues')
axes2[1].set_ylabel('Work Location')
axes2[1].set_title('Mental health Issues')
plt.show()

visual4 = round(subset.groupby('Work_Location')[['Company_Support_for_Remote_Work','Work_Life_Balance_Rating']].mean()*100)
fig3, axes3 = plt.subplots(1,2)
columns = ['Company_Support_for_Remote_Work', 'Work_Life_Balance_Rating']
yvals = ['Remote Work Company Support', 'Work-Life Balance']
for index, (column, yval) in enumerate(zip(columns, yvals)):
    print(column)
    axes3[index].bar(visual4.index, visual4[column], color='green')
    axes3[index].set_title(yval)
    axes3[index].set_ylabel(yval)
    axes3[index].set_xlabel('Work Location')
plt.show()