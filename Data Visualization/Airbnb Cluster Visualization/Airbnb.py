# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)

#Load data
path = r'C:\Users\User\OneDrive\Programació\Python\Kaggle\Airbnb'.replace('\\','/')
data = pd.read_csv(f'{path}/Airbnb_Open_Data.csv')
print(data.head(),'\n')
print(data.info(),'\n')

#Cleaning
data['price'] = data['price'].replace('[\$,.]','', regex=True).astype(float)
data['service fee'] = data['service fee'].replace('[\$,.]','', regex=True).astype(float)
data['last review'] = pd.to_datetime(data['last review'])
data['Total Costs'] = data['price'] + data['service fee']
columnas = [columna.title() for columna in data.columns]
data.columns = columnas
print(data['Neighbourhood Group'].unique(),'\n')
print(data['Cancellation_Policy'].unique(),'\n')
print(data['Room Type'].unique(),'\n')
data.loc[data['Neighbourhood Group']=='brookln','Neighbourhood Group'] = 'Brooklyn'
data.loc[data['Neighbourhood Group']=='manhatan','Neighbourhood Group'] = 'Manhattan'
data['Reviews Per Month'] = data['Reviews Per Month'].fillna(0)

#Visuals
fig_I, ax_I = plt.subplots(1,3)
for col, axis, color in zip(['Total Costs', 'Price', 'Service Fee'], ax_I, ['blue', 'green', 'orange']):
    sns.histplot(data=data[col], linewidth=1, edgecolor='black', color=color, bins=20, ax=axis)
    axis.set_xlabel(f'{col} ($)')
    axis.set_ylabel('Count of Observations')
    axis.set_title(f'{col} distribution')

fig_II, ax_II = plt.subplots(1,3)
for col, axis, color in zip(['Total Costs', 'Price', 'Service Fee'], ax_II, ['blue', 'green', 'orange']):
    sns.boxplot(data = data[col], color=color, ax=axis)
    axis.set_title(f'{col} main statistics')

top_10_prices = data.groupby('Neighbourhood')['Price'].mean().nlargest(10).reset_index()
bottom_10_prices = data.groupby('Neighbourhood')['Price'].mean().nsmallest(10).reset_index()

fig_III, ax_III = plt.subplots(1,2)
fig_III.tight_layout(pad=4)
sns.barplot(data=top_10_prices, x='Neighbourhood', y='Price', edgecolor='Black', linewidth=1, color='darkblue', ax=ax_III[0])
ax_III[0].set_xlabel(None)
ax_III[0].set_ylabel('Price ($)')
ax_III[0].set_title('Top 10 more expensive Neighbourhoods')
ax_III[0].tick_params(axis='x', labelrotation=50)

sns.barplot(data=bottom_10_prices, x='Neighbourhood', y='Price', edgecolor='Black', linewidth=1, color='lightblue', ax=ax_III[1])
ax_III[1].set_xlabel(None)
ax_III[1].set_ylabel('Price ($)')
ax_III[1].set_title('Top 10 cheapest Neighbourhoods')
ax_III[1].tick_params(axis='x', labelrotation=50)

fig_IV, ax_IV = plt.subplots()
sns.scatterplot(data=data, x='Long', y='Lat', hue='Neighbourhood Group', size='Price', ax=ax_IV)
ax_IV.set_xlabel('Longitud (Coordinate)')
ax_IV.set_ylabel('Latitude (Coordinate)')
ax_IV.set_title('Prices Per Area and Neighbourhood')

#Encoding
columns = ['Neighbourhood Group','Instant_Bookable','Cancellation_Policy','Room Type','Construction Year','Total Costs',
            'Minimum Nights','Number Of Reviews','Review Rate Number','Availability 365','Calculated Host Listings Count']
data_enc = data[columns].dropna()

Sc = StandardScaler()
Sc_columns = ['Construction Year','Total Costs','Minimum Nights','Number Of Reviews','Review Rate Number','Calculated Host Listings Count']
data_enc[Sc_columns] = Sc.fit_transform(data_enc[Sc_columns])
Oe = OrdinalEncoder(categories=[['flexible','moderate','strict']])
Oe_columns = ['Cancellation_Policy']
data_enc[Oe_columns] = Oe.fit_transform(data_enc[Oe_columns])
Le = LabelEncoder()
Le_columns = 'Instant_Bookable'
data_enc[Le_columns] = Le.fit_transform(data_enc[Le_columns])
He_columns = ['Room Type','Neighbourhood Group']
data_enc = pd.get_dummies(data=data_enc, columns=He_columns, drop_first=True)

#Principal Component Analysis
pca = PCA(n_components=2)
pca_comp = pca.fit_transform(data_enc)
pca_df = pd.DataFrame(pca_comp, columns=['PCA_I','PCA_II'])
data_pca = data[columns].dropna().reset_index(drop=True)
data_pca['PCA_I'] = pca_df['PCA_I']
data_pca['PCA_II'] = pca_df['PCA_II']

#how many PCAs? Normally until explaining 95% of variance
pca_test = PCA().fit(data_enc)
explained_var = pca_test.explained_variance_ratio_.cumsum()
fig_V, ax_V = plt.subplots()
plt.plot(range(1, len(explained_var) + 1), explained_var, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')

components = pd.DataFrame(pca.components_, columns=data_enc.columns, index=['PCA_I', 'PCA_II']).T
print(components['PCA_I'].nlargest(5),'\n')
print(components['PCA_II'].nlargest(5),'\n')

#Clustering
#How many CLusters? Until innertia drops (elbow) (Inertia is the sum of squared distances from each point to the centroid of the cluster)
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(data_pca[['PCA_I', 'PCA_II']])
    inertia.append(kmeans.inertia_)
fig_VI, ax_VI = plt.subplots()
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method - Optimal Number of Clusters')
plt.grid(True)


kmeans = KMeans(n_clusters=4, random_state=1)
data_pca['Cluster'] = kmeans.fit_predict(data_pca[['PCA_I', 'PCA_II']])
data_pca[['Cancellation_Policy', 'Instant_Bookable']] = data_enc[['Cancellation_Policy', 'Instant_Bookable']]
data_pca = data_pca.loc[(data_pca['Minimum Nights']>=0) & (data_pca['Availability 365']>=0),:]
data_pca = data_pca[(data_pca[data_pca.select_dtypes('number').columns]<data_pca[data_pca.select_dtypes('number').columns].quantile(0.99)).all(axis=1)]

fig_VII, ax_VII = plt.subplots(2,2)
ax_VII = ax_VII.reshape(-1)
sns.scatterplot(data=data_pca, x='PCA_I', y='PCA_II', hue='Cluster', ax=ax_VII[0], palette='rocket')
ax_VII[0].set_title('Clusters and Principal Component Analysis')
ax_VII[0].set_xlabel("PCA I Values")
ax_VII[0].set_ylabel("PCA II Values")

sns.countplot(data=data_pca, x='Cluster', edgecolor='black', linewidth=1, ax=ax_VII[1])
ax_VII[1].set_title('Observations per Cluster')
ax_VII[1].set_ylabel('Number of Observations')
ax_VII[1].set_xlabel('Cluster')

sns.boxplot(data=data_pca, x='Cluster', y='Total Costs', ax=ax_VII[2], color='blue')
ax_VII[2].set_title('Clusters and Total Costs')
ax_VII[2].set_xlabel("Cluster")
ax_VII[2].set_ylabel("Total Costs ($)")

sns.boxplot(data=data_pca, x='Cluster', y='Availability 365', ax=ax_VII[3], color='lightblue')
ax_VII[3].set_title('Clusters and Yearly Availability')
ax_VII[3].set_xlabel("Cluster")
ax_VII[3].set_ylabel("Available 365")

centroids = data_pca[data_pca.select_dtypes('number').columns].groupby('Cluster').mean()
print(centroids,'\n')
print(centroids.std().sort_values(ascending=False),'\n')

fig_VIII, ax_VIII = plt.subplots(1,2)
sns.scatterplot(data=data_pca, x='Availability 365', y='Minimum Nights', hue='Cluster', ax=ax_VIII[0])
ax_VIII[0].set_title('Availability vs Minimum Stay by Cluster')
ax_VIII[0].set_xlabel('Days Available per Year')
ax_VIII[0].set_ylabel('Minimum Nights Required')

sns.scatterplot(data=data_pca, x='Total Costs', y='Number Of Reviews', hue='Cluster', ax=ax_VIII[1])
ax_VIII[1].set_title('Total Costs vs Number of Reviews by Cluster')
ax_VIII[1].set_xlabel('Total Costs ($)')
ax_VIII[1].set_ylabel('Number of Reviews')

sns.relplot(data=data_pca, x='Total Costs', y='Availability 365', hue='Cluster', col='Room Type')

plt.show()

