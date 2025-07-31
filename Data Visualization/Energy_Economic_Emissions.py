#Packages
import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#Download data
path = kagglehub.dataset_download("riazuddinetu/economic-and-globalization-datasets")
print(path,os.listdir(f"{path}/Economic and Globalization Datasets"),'\n')
path = f"{path}/Economic and Globalization Datasets"
data = {}
for i,item in enumerate(os.listdir(path)):
    name = item.split('.')[0]
    data[name] = pd.read_excel(f"{path}/{os.listdir(path)[i]}")

def assign(dict,subdf):
    df = dict[subdf].drop(columns='CountryCode')
    print(f"############ {subdf} ############")
    print(df.head(10),'\n')
    return df
clean_data = assign(data,"1_Clean_Data")
interpole_data = assign(data,"2_Data_interpole")
log_data = assign(data,"3_Data_log_transform")

mapping = {'CO2':'C02 per capita (metric tons)',
            'GI':'Governance Index',
            'EcGI':'Economic Governance Index',
            'SoGI':'Social Governance Index',
            'PoGI':'Political Governance Index',
            'GDP_pc':'GDP per capita (constant 2015 USD)',
            'GDP_pc2':'Squared GDP per capita',
            'EC':'Energy Consumption (kg oil equivalent)',
            'POP':'Population Size',
            'URB': 'Urbanization Rate',
            'REC':'Renewal Energy Consumption (kg oil equivalent)'}

#Data cleaning
clean_data.loc[clean_data['EC'].isna(),'EC'] = interpole_data.loc[clean_data['EC'].isna(),'EC']
clean_data.loc[clean_data['REC'].isna(),'REC'] = interpole_data.loc[clean_data['REC'].isna(),'REC']
clean_data['CO2_pc'] = clean_data['CO2'].copy()
clean_data['CO2'] = clean_data['CO2_pc'].div(1000000000) * clean_data['POP']
clean_data['REC_pc'] = clean_data['REC'].copy()
clean_data['REC'] = clean_data['REC_pc'].div(1000) * clean_data['POP']
clean_data['EC_pc'] = clean_data['EC'].copy()
clean_data['EC'] = clean_data['EC_pc'].div(1000) * clean_data['POP']
clean_data['GDP'] = clean_data['GDP_pc'].div(1000000000) * clean_data['POP']
clean_data['GDP_pc2'] = clean_data['GDP'] ** 2
clean_data['POP'] = clean_data['POP'].div(1000000)
clean_data_gp = clean_data[['Year','CO2','POP','GDP_pc','URB','GI','EcGI','SoGI','PoGI','EC','REC','CO2_pc','REC_pc','GDP','EC_pc']].groupby('Year').agg({'CO2':'sum','POP':'sum','GDP_pc':'mean','URB':'mean','GI':'mean',
                                                                                                                                                          'EcGI':'mean','SoGI':'mean','PoGI':'mean','EC':'sum','REC':'sum',
                                                                                                                                                          'CO2_pc':'mean','REC_pc':'mean','GDP':'sum','EC_pc':'mean'})
clean_data_country = clean_data[['Country','CO2','POP','GDP_pc','URB','GI','EcGI','SoGI','PoGI','EC','REC','CO2_pc','REC_pc','GDP','EC_pc']].groupby('Country').agg({'CO2':'sum','POP':'sum','GDP_pc':'mean','URB':'mean','GI':'mean',
                                                                                                                                                          'EcGI':'mean','SoGI':'mean','PoGI':'mean','EC':'sum','REC':'sum',
                                                                                                                                                          'CO2_pc':'mean','REC_pc':'mean','GDP':'sum','EC_pc':'mean'})
#Visualization
fig_I, axes_I = plt.subplots(4,1)
fig_I.tight_layout(pad=1)
axes_I = axes_I.flatten()
sns.lineplot(data=clean_data_gp, x='Year', y='GDP', ax=axes_I[0], marker='o',markeredgecolor='black', label='GDP (USB $)')
axes_I[0].set_xticks(clean_data_gp.index)
axes_I[0].set_xticklabels(clean_data_gp.index)
axes_I[0].set_xlabel(None)
axes_I[0].set_ylabel('Billions US Dollars')
axes_I[0].set_title('GDP and C02 Emissions evolution')
axes_I[0].set_ylim(clean_data_gp['GDP'].min()*0.9, clean_data_gp['GDP'].max()*1.10)
axes_IB = axes_I[0].twinx()
sns.lineplot(x=clean_data_gp.index, y=clean_data_gp['CO2'], ax=axes_IB,marker='o',markeredgecolor='black', label='C02 (Metric Gigatons)', color='crimson')
axes_IB.set_xlabel(None)
axes_IB.set_ylabel('C02 Metric Gigatons')
axes_IB.set_ylim(clean_data_gp['CO2'].min()*0.9, clean_data_gp['CO2'].max()*1.10)
axes_I[0].legend(loc='upper left')

sns.lineplot(x=clean_data_gp.index, y=clean_data_gp['POP'], ax=axes_I[1], marker='o',markeredgecolor='black', label='Population (Millions)', color='burlywood')
axes_I[1].set_xticks(clean_data_gp.index)
axes_I[1].set_xticklabels(clean_data_gp.index)
axes_I[1].set_xlabel(None)
axes_I[1].set_ylabel('World Population Millions')
axes_I[1].set_title('Global Population and Urbanization rate evolution')
axes_I[1].set_ylim((clean_data_gp['POP']).min()*0.9, (clean_data_gp['POP']).max()*1.10)
axes_IC = axes_I[1].twinx()
sns.lineplot(x=clean_data_gp.index, y=clean_data_gp['URB'], ax=axes_IC,marker='o',markeredgecolor='black', label='Urbanization Rate', color='darkslategray')
axes_IC.set_xlabel(None)
axes_IC.set_ylabel('Urbanization Rate (%)')
axes_IC.set_ylim(clean_data_gp['URB'].min()*0.9, clean_data_gp['URB'].max()*1.10)
axes_I[1].legend(loc='upper left')

sns.lineplot(data=clean_data_gp, x='Year', y='GDP_pc', ax=axes_I[2], marker='o',markeredgecolor='black', label='GDP per capita (US $)', color='navy')
axes_I[2].set_xticks(clean_data_gp.index)
axes_I[2].set_xticklabels(clean_data_gp.index)
axes_I[2].set_xlabel(None)
axes_I[2].set_ylabel('GDP per capita')
axes_I[2].set_title('GDP per capita and C02 Emissions per capita evolution')
axes_I[2].set_ylim(clean_data_gp['GDP_pc'].min()*0.9, clean_data_gp['GDP_pc'].max()*1.10)
axes_ID = axes_I[2].twinx()
sns.lineplot(x=clean_data_gp.index, y=clean_data_gp['CO2_pc'], ax=axes_ID, marker='o',markeredgecolor='black', label='C02 per capita', color='darkred')
axes_ID.set_xlabel(None)
axes_ID.set_ylabel('CO2 per capita (Tons)')
axes_ID.set_ylim(clean_data_gp['CO2_pc'].min()*0.9, clean_data_gp['CO2_pc'].max()*1.10)
axes_I[2].legend(loc='upper left')

sns.lineplot(x=clean_data_gp.index, y=clean_data_gp['EC_pc']/1000, ax=axes_I[3], marker='o', markeredgecolor='black', label='Energy Consumption pc', color='orange')
axes_I[3].set_xticks(clean_data_gp.index)
axes_I[3].set_xticklabels(clean_data_gp.index)
axes_I[3].set_xlabel(None)
axes_I[3].set_ylabel('Metric Tons of oil equivalent')
axes_I[3].set_title('Renewal and Total Energy Consumption per capita')
axes_I[3].set_ylim((clean_data_gp['EC_pc']/1000).min()*0.9, (clean_data_gp['EC_pc']/1000).max()*1.10)
axes_IE = axes_I[3].twinx()
sns.lineplot(x=clean_data_gp.index, y=clean_data_gp['REC_pc'], ax=axes_IE, marker='o', markeredgecolor='black', label='Renewal Energy Consumption pc', color='forestgreen')
axes_IE.set_ylabel('Metric Kg of oil equivalent')
axes_IE.set_ylim(clean_data_gp['REC_pc'].min()*0.9, clean_data_gp['REC_pc'].max()*1.10)
axes_I[3].legend(loc='upper left')

corr = clean_data_gp.reset_index(drop=True).drop(columns=['GI','GDP_pc','CO2_pc']).corr()
fig_II, axes_II = plt.subplots()
sns.heatmap(data=corr, annot=True, cmap='RdBu', edgecolor='black',linecolor='black', linewidth=0.65)
axes_II.set_title('Correlation of main variables')
axes_II.tick_params(axis='x', rotation=0)

top_co2, top_gdp, top_ec, top_rec = [clean_data.loc[:,['Country',column, f"{column}_pc"]].groupby('Country').agg({column:'sum',f"{column}_pc":'mean'}) for column in ['CO2','GDP','EC','REC']]
top_ec.iloc[:,0] = top_ec.iloc[:,0].div(1000000000)
top_rec.iloc[:,0] = top_rec.iloc[:,0].div(1000000000)
map_title = {'CO2':'CO2 Emissions',
             'GDP':'Gros Domestic Product',
             'EC':'Energy Consumption',
             'REC':'Renewable Energy Consumption',
             'CO2_pc':'CO2 Emissions per capita',
             'GDP_pc':'Gros Domestic Product per capita',
             'EC_pc':'Energy Consumption per capita',
             'REC_pc':'Renewable Energy Consumption per capita'}
map_label = {'CO2':'C02 Metric Gigatons',
             'GDP':'constant 2015 USD Billions',
             'EC':'Metric Gigatonnes of oil equivalent',
             'REC':'Metric Gigatonnes of oil equivalent',
             'CO2_pc':'CO2 Metric Tons',
             'GDP_pc':'Constant 2015 USD',
             'EC_pc':'Metric Kg of oil equivalent',
             'REC_pc':'Metric Kg of oil equivalent'}
fig_III, axes_III = plt.subplots(4,2)
fig_III.tight_layout()
axes_III = axes_III.flatten()
colors = ['darkred','lightcoral','navy','lightskyblue','darkorange','sandybrown','darkgreen','mediumseagreen']
for i, df in enumerate([top_co2, top_gdp, top_ec, top_rec]):
    title_I = map_title[df.columns[0]]
    title_II = map_title[df.columns[1]]
    label_I = map_label[df.columns[0]]
    label_II = map_label[df.columns[1]]
    data_I = df.nlargest(5,df.columns[0])
    data_II = df.nlargest(5,df.columns[1])
    sns.barplot(x=data_I.index, y=data_I.iloc[:,0], ax=axes_III[2*i], linewidth=1,edgecolor='black', color=colors[2*i])
    sns.barplot(x=data_II.index, y=data_II.iloc[:,1], ax=axes_III[2*i+1], linewidth=1,edgecolor='black', color=colors[2*i+1])
    axes_III[2*i].set_xlabel(None)
    axes_III[2*i].set_ylabel(label_I)
    axes_III[2*i].set_title(f'Top 5 Countries by {title_I}')
    axes_III[2*i+1].set_xlabel(None)
    axes_III[2*i+1].set_ylabel(label_II)
    axes_III[2*i+1].set_title(f'Top 5 Countries by {title_II}')

data_top_5 = clean_data.loc[clean_data['Country'].isin(top_gdp.nlargest(5,'GDP').index),['Country','Year','GDP','CO2','EC','REC','POP']].melt(id_vars=['Country','Year'], value_vars=['GDP','CO2','EC','REC','POP'], var_name='Indicator', value_name='Value')
data_top_5_pc = clean_data.loc[clean_data['Country'].isin(top_gdp.nlargest(5,'GDP').index),['Country','Year','GDP_pc','CO2_pc','EC_pc','REC_pc']].melt(id_vars=['Country','Year'], value_vars=['GDP_pc','CO2_pc','EC_pc','REC_pc'], var_name='Indicator', value_name='Value')

fig_IV, axes_IV = plt.subplots(2,2)
fig_IV.tight_layout()
axes_IV = axes_IV.flatten()
sns.lineplot(data=data_top_5.loc[data_top_5['Indicator']=='GDP',:], x='Year', y='Value', hue='Country', marker='o', markeredgecolor='black', dashes=False, linewidth=1.5, ax=axes_IV[0])
axes_IV[0].set_xlabel(None)
axes_IV[0].set_ylabel('GDP Billions US$')
axes_IV[0].set_title('GDP and CO2 Emissions for top 5 economies')
axes_IV[0].legend(loc='upper left')
axes_IVB = axes_IV[0].twinx()
sns.lineplot(data=data_top_5.loc[data_top_5['Indicator']=='CO2',:], x='Year', y='Value', hue='Country', marker='.', markeredgecolor='black', dashes=(2,2), linewidth=1.5, ax=axes_IVB)
axes_IVB.set_ylabel('CO2 Emissions Gigatonnes')
axes_IVB.legend(loc='upper right')

sns.lineplot(data=data_top_5_pc.loc[data_top_5_pc['Indicator']=='GDP_pc',:], x='Year', y='Value', hue='Country', marker='d', markeredgecolor='black', dashes=False, linewidth=1.5, ax=axes_IV[1])
axes_IV[1].set_xlabel(None)
axes_IV[1].set_ylabel('GDP Constant 2015 US$')
axes_IV[1].set_title('GDP and CO2 Emissions per capita for top 5 economies')
axes_IV[1].legend(loc='upper left')
axes_IVC = axes_IV[1].twinx()
sns.lineplot(data=data_top_5_pc.loc[data_top_5_pc['Indicator']=='CO2_pc',:], x='Year', y='Value', hue='Country', marker='.', markeredgecolor='black', dashes=(2,2), linewidth=1.5, ax=axes_IVC)
axes_IVC.set_ylabel('CO2 Emissions Tonnes')
axes_IVC.legend(loc='upper right')

data_top_5.loc[data_top_5['Indicator'].isin(['EC','REC']),'Value'] = data_top_5.loc[data_top_5['Indicator'].isin(['EC','REC']),'Value'].div(1000000)
sns.lineplot(data=data_top_5.loc[data_top_5['Indicator']=='EC',:], x='Year', y='Value', hue='Country', marker='p', markeredgecolor='black', dashes=False, linewidth=1.5, ax=axes_IV[2])
axes_IV[2].set_xlabel(None)
axes_IV[2].set_ylabel('Energy Consumption Gigatonnes of oil equivalent')
axes_IV[2].set_title('Total and Renewable energy consumption for top 5 economies')
axes_IV[2].legend(loc='upper left')
axes_IVD = axes_IV[2].twinx()
sns.lineplot(data=data_top_5.loc[data_top_5['Indicator']=='REC',:], x='Year', y='Value', hue='Country', marker='.', markeredgecolor='black', dashes=(2,2), linewidth=1.5, ax=axes_IVD)
axes_IVD.set_ylabel('Renewable Energy Consumption Gigatonnes of oil equivalent')
axes_IVD.legend(loc='upper right')

sns.lineplot(data=data_top_5_pc.loc[data_top_5_pc['Indicator']=='EC_pc',:], x='Year', y='Value', hue='Country', marker='v', markeredgecolor='black', dashes=False, linewidth=1.5, ax=axes_IV[3])
axes_IV[3].set_xlabel(None)
axes_IV[3].set_ylabel('Energy Consumption per Kg of oil equivalent')
axes_IV[3].set_title('Total and Renewable energy consumption per capita for top 5 economies')
axes_IV[3].legend(loc='upper left')
axes_IVE = axes_IV[3].twinx()
sns.lineplot(data=data_top_5_pc.loc[data_top_5_pc['Indicator']=='REC_pc',:], x='Year', y='Value', hue='Country', marker='.', markeredgecolor='black', dashes=(2,2), linewidth=1.5, ax=axes_IVE)
axes_IVE.set_ylabel('Renewable Energy Consumption Kg of oil equivalent')
axes_IVE.legend(loc='upper right')

years = [2010, 2015, 2020]
fig_V, axes_V = plt.subplots(1,3,subplot_kw=dict(polar=True))
fig_V.tight_layout()
fig_V.suptitle('Features Comparison for the top 5 Economies')
axes_V = axes_V.flatten()
for year,ax in zip(years,axes_V):
    radar_data = clean_data.loc[clean_data['Year']==year,['Country','CO2','POP','GDP_pc','URB','GI','EcGI','SoGI','PoGI','EC','REC','CO2_pc','REC_pc','GDP','EC_pc']].set_index('Country')
    Countries = radar_data.nlargest(5,'GDP').index
    Indicators = ['GDP_pc','CO2_pc','EcGI','SoGI','PoGI','REC_pc']
    radar_data = radar_data.loc[radar_data.index.isin(Countries),Indicators]
    mMs = MinMaxScaler(feature_range=(0.05, 0.95))
    radar_data = mMs.fit_transform(radar_data.loc[Countries])
    angles = np.linspace(0,2*np.pi,len(Indicators), endpoint=False).tolist()
    angles += angles[:1]
    scaled_rows = [list(row) + [row[0]] for row in radar_data]
    for stats, country in zip(scaled_rows,Countries):
        ax.plot(angles,stats,label=country, linewidth=2)
        ax.fill(angles,stats,alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(Indicators)
    ax.set_title(f'GDP and CO2 per capita and Governance Indices for the year {year}', y=1.1)
    ax.legend(bbox_to_anchor = (0.04, 1))

fig_VI, axes_VI = plt.subplots(2,2)
fig_VI.tight_layout()
axes_VI = axes_VI.flatten()
scatter_data_abs = clean_data.loc[clean_data['Country'].isin(clean_data_country.nlargest(10,'GDP').index),['Country','Year','CO2','POP','GDP_pc','URB','GI','EcGI','SoGI','PoGI','EC','REC','CO2_pc','REC_pc','GDP','EC_pc']]
scatter_data_abs[['EC','REC']] = scatter_data_abs[['EC','REC']].div(1000000)
scatter_data_pc = clean_data.loc[clean_data['Country'].isin(clean_data_country.nlargest(10,'GDP_pc').index),['Country','CO2','POP','GDP_pc','URB','GI','EcGI','SoGI','PoGI','EC','REC','CO2_pc','REC_pc','GDP','EC_pc']]
sns.scatterplot( x=scatter_data_abs['GDP'],y=scatter_data_abs['REC'],edgecolor='black', ax=axes_VI[0], hue=scatter_data_abs['Country'])
axes_VI[0].set_xlabel('GDP Billions US$')
axes_VI[0].set_ylabel('Renewable Energy Consumption Gigatonnes of oil equivalent')
axes_VI[0].set_title('GDP and Renewable Energy Consumption Relation for top 10 Economies')

sns.scatterplot( x=scatter_data_abs['GDP_pc'],y=scatter_data_abs['CO2_pc'],edgecolor='black', ax=axes_VI[1], hue=scatter_data_abs['Country'])
axes_VI[1].set_xlabel('GDP 2015 Constant US$')
axes_VI[1].set_ylabel('CO2 Emissions Gigatonnes')
axes_VI[1].set_title('GDP and CO2 Emissions per capita Relation for top 10 Economies')

sns.scatterplot( x=scatter_data_abs['EC_pc'],y=scatter_data_abs['REC_pc'],edgecolor='black', ax=axes_VI[2], hue=scatter_data_abs['Country'])
axes_VI[2].set_xlabel('Energy Consumption Kg of oil equivalent')
axes_VI[2].set_ylabel('Renewable Energy Consumption Kg of oil equivalent')
axes_VI[2].set_title('Total and Renewable energy consumption per capita relation for top 10 Economies')

sns.scatterplot( x=scatter_data_abs['CO2_pc'],y=scatter_data_abs['URB'],edgecolor='black', ax=axes_VI[3], hue=scatter_data_abs['Country'])
axes_VI[3].set_xlabel('Tonnes of CO2')
axes_VI[3].set_ylabel('Urbanization rate (%)')
axes_VI[3].set_title('CO2 Emissions per capita and Urbanization rate for top 10 Economies')

plt.show()
