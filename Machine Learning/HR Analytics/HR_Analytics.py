import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pd.set_option('display.max_columns',None)

# Download latest version
path = r'C:\Users\User\OneDrive\Programació\Python\Kaggle\HR Analytics'

print("Path to dataset files:", path)
print(os.listdir(path=path), '\n')

data = pd.read_csv(f'{path.replace('\\','/')}/WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(data.head(),'\n')

#Some analytics
attrition_characteristics = data.groupby('Attrition')[['Age','TotalWorkingYears','JobSatisfaction','YearsSinceLastPromotion','YearsWithCurrManager']].median()
print(attrition_characteristics,'\n')
for item in ['OverTime','Department','Gender','EducationField']:
    print(data.loc[data['Attrition']=='Yes', item].value_counts(),'\n')

#Visualization
fig_I, axes_I = plt.subplots(2,2)
fig_I.tight_layout(pad=2)
fig_I.suptitle('Absolute Frequency of Attrition')
axes = [axes_I[0,0],axes_I[0,1],axes_I[1,0],axes_I[1,1]]
for var, ax in zip(['OverTime','Department','Gender','EducationField'], axes):
    sns.countplot(data=data,hue='Attrition',x=var,ax=ax, edgecolor='black', linewidth=1)
    ax.set_title(f'Countplot of {var}')
    ax.set_ylabel('Number of Employees')
    ax.set_xlabel(None)
    plt.xticks(rotation=30)


fig_II, axes_II = plt.subplots(2,2)
fig_II.tight_layout(pad=2)
fig_II.suptitle('Relative Frequency of Attrition')
axes = [axes_II[0,0],axes_II[0,1],axes_II[1,0],axes_II[1,1]]
for var, ax in zip(['OverTime','Department','Gender','EducationField'], axes):
    sns.histplot(data=data, x=var, hue='Attrition', stat='percent', common_norm=False, multiple='dodge', edgecolor='black', linewidth=1, ax=ax)
    ax.set_title(f'Countplot of {var}')
    ax.set_ylabel('Percentage of Employees')
    ax.set_xlabel(None)
    plt.xticks(rotation=30)

#Preprocessing
Le = LabelEncoder()
Sc = StandardScaler()
data_clean = data.drop(columns=['Attrition','EmployeeCount','EmployeeNumber'])
for col in data_clean.select_dtypes('object').columns:
    data_clean[col] = Le.fit_transform(data_clean[col])

data_clean = Sc.fit_transform(data_clean)

#Principal Component Analysis (PCA)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(data_clean)
pca_frame = pd.DataFrame(pca_components,columns=['PCA_I', 'PCA_II'])
print(pca_frame.head(),'\n')

#Mean Clustering
kmeans = KMeans(n_clusters=2, random_state=1)
clusters = kmeans.fit_predict(pca_components)

#Data included 
data['Cluster'] = clusters
data['PCA_I'] = pca_frame['PCA_I']
data['PCA_II'] = pca_frame['PCA_II']

#Cluster Visuals
fig_III, axes_III = plt.subplots(1,2)
fig_III.tight_layout(pad=2)
sns.scatterplot(data=data, x='PCA_I', y='PCA_II', hue='Attrition', ax=axes_III[0])
axes_III[0].set_xlabel('Principal Component I')
axes_III[0].set_ylabel('Principal Component II')
axes_III[0].set_title('Principal Components by Attrition')
sns.scatterplot(data=data, x='PCA_I', y='PCA_II', hue='Cluster', ax=axes_III[1])
axes_III[1].set_xlabel('Principal Component I')
axes_III[1].set_ylabel('Principal Component II')
axes_III[1].set_title('Principal Components by Cluster')

plt.show()

#Clusters and PCA info
pca_composition = pd.DataFrame(pca.components_, columns=data.drop(columns=['Attrition','EmployeeCount','EmployeeNumber','Cluster','PCA_I','PCA_II']).columns, index=['PCA_I','PCA_II']).T
print(pca_composition.sort_values(by=['PCA_I','PCA_II'], ascending=[False,False]),'\n')

centroids_pca = kmeans.cluster_centers_
centroids_scaled = pca.inverse_transform(centroids_pca)
centroids_original = Sc.inverse_transform(centroids_scaled)
centroid_df = pd.DataFrame(centroids_original, columns=data.drop(columns=['Attrition','EmployeeCount','EmployeeNumber','Cluster','PCA_I','PCA_II']).columns)
print(centroid_df,'\n')
print(pd.crosstab(data['Cluster'],data['Attrition'], normalize='index')*100,'\n')