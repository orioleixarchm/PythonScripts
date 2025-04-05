#Importing packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns',None)

#Load data
path = r'C:\Users\User\OneDrive\Programació\Python\Kaggle\Exercise_health'.replace('\\','/')
print(os.listdir(path),'\n')
file = 'gym_members_exercise_tracking'
data = pd.read_csv(f'{path}/{file}.csv')
print(data.head())

##Visuals
#Demographics and Distributions
fig, axes = plt.subplots(2,2)
fig.suptitle('Demographics & Distributions')
fig.tight_layout(pad=0.2)
sns.histplot(data=data, x='Age', kde=True, bins=20, ax=axes[0,0])
axes[0,0].set_title('Age Distribution')
axes[0,0].set_xlabel(None)
axes[0,0].set_ylabel('Nº Observations')
sns.histplot(data=data, x='BMI', kde=True, bins=20, hue='Gender', ax=axes[0,1])
axes[0,1].set_title('BMI Distribution by Gender')
axes[0,1].set_xlabel(None)
axes[0,1].set_ylabel('Nº Observations')
sns.countplot(data=data, x='Workout_Type', hue='Gender', ax=axes[1,0], edgecolor='black', linewidth=0.8)
axes[1,0].set_title('Workout Populatiry per Gender')
axes[1,0].set_xlabel(None)
axes[1,0].set_ylabel('Nº Observations')
sns.boxplot(data=data, x='Experience_Level', y='Gender', ax=axes[1,1])
axes[1,1].set_title('Experience Level by Gender')
axes[1,1].set_xlabel(None)
axes[1,1].set_ylabel(None)

#Body Composition & Health Stats
figI, axesI_A = plt.subplots(1,3)
figI.suptitle('Body Composition & Health Stats')
figI.tight_layout(pad=1.5)
sns.regplot(data=data, x='Workout_Frequency (days/week)', y='Fat_Percentage', ax=axesI_A[0]) #Do more frequent workouts reduce fat percentage?
axesI_A[0].set_title(r'Fat % vs Training days per week and Hours per training')
axesI_A[0].set_xlabel('Trainings per week')
axesI_A[0].set_ylabel('Fat Percentage (%)')
sns.scatterplot(data=data, x='Session_Duration (hours)', y='Fat_Percentage', hue='Gender', ax=axesI_A[1]) #Do longer workouts reduce fat percentage?
axesI_A[1].set_xlabel('Hours per Training')
sns.scatterplot(data=data, x='BMI', y='Calories_Burned', hue='Gender', ax=axesI_A[2]) #Do longer workouts reduce fat percentage?
axesI_A[2].set_title('Calories burned vs BMI')
axesI_A[2].set_xlabel('BMI Index')
axesI_A[2].set_ylabel('Calories Burned (Kcal)')

figII, axesII = plt.subplots(1,3)
figII.suptitle('Body Composition & Health Stats')
figII.tight_layout(pad=1.5)
sns.boxplot(data=data, x='Resting_BPM', y='Gender', ax=axesII[0])
axesII[0].set_title('Resting Beats per minute (BPM) per experience Level')
axesII[0].set_xlabel('Resting Beats per minute (BPM)')
axesII[0].set_ylabel(None)
sns.boxplot(data=data, x='Workout_Type', y='Fat_Percentage', hue='Gender', ax=axesII[1])
axesII[1].set_title('Fat percentage per Workout')
axesII[1].set_xlabel(None)
axesII[1].set_ylabel('Fat percentage (%)')
sns.scatterplot(data=data, x='BMI', y='Fat_Percentage', hue='Gender', ax=axesII[2])
axesII[2].set_title('Fat Percentage per Body Mass Index')
axesII[2].set_xlabel('BMI Index')
axesII[2].set_ylabel('Fat percentage (%)')

sns.pairplot(data=data, vars=['Fat_Percentage', 'Water_Intake (liters)', 'Calories_Burned', 'BMI'], hue='Gender')

#Performance & Workout Output
figIII, axesIII = plt.subplots(1,3)
figIII.suptitle('Performance & Workout Output')
figIII.tight_layout(pad=1.5)
sns.boxplot(data=data, y='Calories_Burned', x='Workout_Type', hue='Gender', ax=axesIII[0])
axesIII[0].set_title('Calories burned per workout')
axesIII[0].set_xlabel(None)
axesIII[0].set_ylabel('Calories Burned')
sns.regplot(data=data, y='Calories_Burned', x='Session_Duration (hours)', color='Green', ax=axesIII[1])
sns.scatterplot(data=data, y='Calories_Burned', x='Session_Duration (hours)', hue='Gender', ax=axesIII[1])
axesIII[1].set_title('Calories Burned per Workout duration')
axesIII[1].set_xlabel(None)
axesIII[1].set_ylabel('Calories Burned')
sns.regplot(data=data, y='Calories_Burned', x='Workout_Frequency (days/week)', ax=axesIII[2])
axesIII[2].set_title('Calories Burned per Workout weekly frequency')
axesIII[2].set_xlabel(None)
axesIII[2].set_ylabel('Calories Burned')

#Hydration & Behavior Patterns
figIV, axesIV = plt.subplots(1,3)
figIV.suptitle('Hydration & Behavior Patterns')
figIV.tight_layout(pad=1.5)
sns.boxplot(data=data, y='Water_Intake (liters)', x='Workout_Type', hue='Gender', ax=axesIV[0])
axesIV[0].set_title('Water Intake burned per workout')
axesIV[0].set_xlabel(None)
axesIV[0].set_ylabel('Liters of Water')
sns.regplot(data=data, y='Water_Intake (liters)', x='BMI', color='Green', ax=axesIV[1])
sns.scatterplot(data=data, y='Water_Intake (liters)', x='BMI', hue='Gender', ax=axesIV[1])
axesIV[1].set_title('Calories Burned per Workout duration')
axesIV[1].set_xlabel('Body Mass Index')
axesIV[1].set_ylabel('Liters of Water')
sns.violinplot(data=data, y='Water_Intake (liters)', x='Workout_Frequency (days/week)', ax=axesIV[2])
axesIV[2].set_title('Water Intake per Workout weekly frequency')
axesIV[2].set_xlabel(None)
axesIV[2].set_ylabel('Liters of Water')

#Experience & Fitness Analysis
bpm_data = data[['Experience_Level','Max_BPM','Avg_BPM','Resting_BPM']].melt(id_vars='Experience_Level', value_vars=['Max_BPM','Avg_BPM','Resting_BPM'],
                                                                             var_name='BPM_Type', value_name='BPM_Values')
figV, axesV = plt.subplots(2,2)
figV.suptitle('Experience & Fitness Analysis')
figV.tight_layout(pad=0.8)
sns.boxplot(data=bpm_data, y='BPM_Values', x='Experience_Level', hue='BPM_Type', ax=axesV[0,0])
axesV[0,0].set_title('Beats Per Minute values per Experience Level')
axesV[0,0].set_xlabel('Experience Level')
axesV[0,0].set_ylabel('BPM')
sns.boxplot(data=data, y='Calories_Burned', x='Experience_Level', hue='Gender', ax=axesV[0,1])
axesV[0,1].set_title('Calories Burned per Experience Level')
axesV[0,1].set_xlabel('Experience Level')
axesV[0,1].set_ylabel('Calories Burned')
sns.barplot(data=data, y='Session_Duration (hours)', x='Experience_Level', hue='Gender', ax=axesV[1,0], edgecolor='black', linewidth=0.8)
axesV[1,0].set_title('Average Workout length per Experience Level')
axesV[1,0].set_ylabel('Experience Level')
axesV[1,0].set_xlabel('Hours of training')
sns.barplot(data=data, y='Workout_Frequency (days/week)', x='Experience_Level', hue='Gender', ax=axesV[1,1], edgecolor='black',  linewidth=0.8)
axesV[1,1].set_title('Weekly workout frequency per Experience Level')
axesV[1,1].set_ylabel('Experience Level')
axesV[1,1].set_xlabel('Trainings per week')

#Correlation, PCA & Clustering
continuous_data = data.loc[:,data.nunique()>6]
Sc = StandardScaler()
continuous_data_sc = Sc.fit_transform(continuous_data)
mycolors = ['#1f77b4', '#ff7f0e', '#2ca02c']  #Blue, Orange, Green color palette

figVI, axesVI = plt.subplots(2,2)
figVI.suptitle('Correlation & Clustering')
figVI.tight_layout(pad=0.8)
cluster = KMeans(n_clusters=3, random_state=1)
clusters = cluster.fit_predict(continuous_data_sc)
continuous_data['Cluster'] = clusters

sns.scatterplot(data=continuous_data, y='Calories_Burned', x='Fat_Percentage', hue='Cluster', ax=axesVI[0,0], palette=mycolors)
axesVI[0,0].set_title('Calories Burned and Fat Percentage Clustered')
axesVI[0,0].set_xlabel('Fat Percentage (%)')
axesVI[0,0].set_ylabel('Calories Burned')
sns.scatterplot(data=data, y='Calories_Burned', x='Fat_Percentage', hue='Experience_Level', ax=axesVI[0,1], palette=mycolors)
axesVI[0,1].set_title('Calories Burned and Fat Percentage per Experience Level')
axesVI[0,1].set_xlabel('Fat Percentage (%)')
axesVI[0,1].set_ylabel('Calories Burned')

pca = PCA(n_components=2)
pcas = pca.fit_transform(continuous_data_sc)
pca_df = pd.DataFrame(pcas, columns=['PCA_I', 'PCA_II'])
pca_df['Cluster'] = clusters
pca_df['Gender'] = data['Gender'] 

sns.scatterplot(data=pca_df, y='PCA_I', x='PCA_II', hue='Cluster', ax=axesVI[1,0], palette=mycolors)
axesVI[1,0].set_title('PCA I and PCA II Clustered')
axesVI[1,0].set_xlabel('PCA II')
axesVI[1,0].set_ylabel('PCA I')
sns.regplot(data=pca_df, x='PCA_II', y='PCA_I', ax=axesVI[1,1], color='Green')
sns.scatterplot(data=pca_df, x='PCA_II', y='PCA_I', hue='Gender')
axesVI[1,1].set_title('PCA I and PCA II per Gender')
axesVI[1,1].set_xlabel('Fat Percentage (%)')
axesVI[1,1].set_ylabel('Calories Burned')

figVII, axesVI = plt.subplots()
sns.heatmap(data=continuous_data.corr(), annot=True, cmap='Blues', linewidths=0.4, linecolor='black')
plt.title('Correlation of Main Contious variables')
plt.xticks(rotation=25)
plt.show()