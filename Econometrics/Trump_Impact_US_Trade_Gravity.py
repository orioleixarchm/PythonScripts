#Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import comtradeapicall
import wbdata
from fredapi import Fred

#Paramters Setting
years = ["2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012",
         "2013","2014","2015","2016","2017","2018","2019","2020","2021","2022","2023","2024"]
reporter_code = "842"
partner_code = ["156", "276", "392", "484", "124", "380", "76"]
code_name_map = {"842":"USA", "156":"China", "276":"Germany", "392":"Japan",
                 "484":"Mexico", "124":"Canada", "380":"Italy", "76":"Brazil"}

api_key = 'f10516bd8938e6e62aae398d7394b76f'
fred = Fred(api_key=api_key)

indicators = {
    "NY.GDP.MKTP.KD": "GDP"
}

#Data Retrieval
def trade_flows(reporter, partner, year_list, flow="X"):
    df = {}
    for id in partner:
        data = pd.DataFrame()
        for year in year_list:
            data_1 = comtradeapicall.getFinalData(
                None,  # No pre-fetched data
                "C",   # typeCode: Commodity data
                "A",   # freqCode: Annual data Can also be M for monthly
                "HS",  # clCode: Harmonized System classification
                year,  # period: Years stated
                reporter,   # reporterCode: reporter code
                "TOTAL", # cmdCode: All commodities
                flow,     # flowCode: Exports (M = Imports, X = Exports)
                id,   # partnerCode: partner codes
                "",   # partner2Code: Default (empty for all)
                "",      # customsCode: Default (empty for all)
                ""       # motCode: Default (empty for all)
            )[['refYear', 'reporterCode', 'partnerCode', 'primaryValue']]
            data = pd.concat([data,data_1])
        df[code_name_map[id]] = data
        print(code_name_map[id], flow)
        print(df[code_name_map[id]].head(),'\n')
    return df

DFX = trade_flows(reporter=reporter_code, partner=partner_code, year_list=years, flow="X")
DFM = trade_flows(reporter=reporter_code, partner=partner_code, year_list=years, flow="M")

Crisis = fred.get_series('USRECQ', observation_start='01-01-2000').resample('YE').max().astype(int)
Crisis.index = Crisis.index.year.astype(int)
Crisis.name = 'Crisis'

trade_data = wbdata.get_dataframe(indicators, country=["JPN", "DEU", "USA", "CHN", "MEX", "CAN", "ITA", "BRA"]).reset_index(drop=False)
trade_data['date'] = trade_data['date'].astype(int)

#Data manipulation and merge
def data_enrichment(country, flow="X"):
    if flow == "X":
        df = DFX[country].set_index('refYear')
        df['reporterCode'] = df['reporterCode'].astype('str').map(code_name_map)
        df['partnerCode'] = df['partnerCode'].astype('str').map(code_name_map)
        df = df.join(Crisis, how='left')
        df = df.merge(trade_data.loc[trade_data['country']=="United States",['date','GDP']], 
                                    left_index=True, right_on='date').rename(columns={'GDP':'Exporter_GDP','reporterCode':'Exporter',
                                    'partnerCode':'Importer','primaryValue':'Trade_flow'}).set_index('date')
        df = df.merge(trade_data.loc[trade_data['country']==country,['date','GDP']], 
                                left_index=True, right_on='date').rename(columns={'GDP':'Importer_GDP'}).set_index('date')
    else:
        df = DFM[country].set_index('refYear')
        df['reporterCode'] = df['reporterCode'].astype('str').map(code_name_map)
        df['partnerCode'] = df['partnerCode'].astype('str').map(code_name_map)
        df = df.join(Crisis, how='left')
        df = df.merge(trade_data.loc[trade_data['country']=="United States",['date','GDP']], 
                                    left_index=True, right_on='date').rename(columns={'GDP':'Importer_GDP','reporterCode':'Importer',
                                    'partnerCode':'Exporter','primaryValue':'Trade_flow'}).set_index('date')
        df = df.merge(trade_data.loc[trade_data['country']==country,['date','GDP']], 
                                left_index=True, right_on='date').rename(columns={'GDP':'Exporter_GDP'}).set_index('date')
    return df

US_China_X = data_enrichment('China', flow="X")
US_Japan_X = data_enrichment('Japan', flow="X")
US_Germany_X = data_enrichment('Germany', flow="X")
US_Mexico_X = data_enrichment('Mexico', flow="X")
US_Canada_X = data_enrichment('Canada', flow="X")
US_Italy_X = data_enrichment('Italy', flow="X")
US_Brazil_X = data_enrichment('Brazil', flow="X")

US_China_M = data_enrichment('China', flow="M")
US_Japan_M = data_enrichment('Japan', flow="M")
US_Germany_M = data_enrichment('Germany', flow="M")
US_Mexico_M = data_enrichment('Mexico', flow="M")
US_Canada_M = data_enrichment('Canada', flow="M")
US_Italy_M = data_enrichment('Italy', flow="M")
US_Brazil_M = data_enrichment('Brazil', flow="M")

data_all = pd.concat([US_China_X.reset_index(drop=False),US_Japan_X.reset_index(drop=False),US_Germany_X.reset_index(drop=False),
                      US_Mexico_X.reset_index(drop=False),US_Canada_X.reset_index(drop=False),US_Italy_X.reset_index(drop=False),
                      US_Brazil_X.reset_index(drop=False), US_China_M.reset_index(drop=False), US_Japan_M.reset_index(drop=False),
                      US_Germany_M.reset_index(drop=False),US_Mexico_M.reset_index(drop=False), US_Canada_M.reset_index(drop=False),
                      US_Italy_M.reset_index(drop=False),US_Brazil_M.reset_index(drop=False)], axis=0)

#Visualization
data_vis = data_all.copy()
data_vis['Trade_flow'] = data_vis['Trade_flow'].div(1000000000)
data_vis = {'X':data_vis.loc[data_vis['Exporter']=='USA',:].pivot(index='date', columns='Importer', values='Trade_flow'),
            'M':data_vis.loc[data_vis['Importer']=='USA',:].pivot(index='date', columns='Exporter', values='Trade_flow')}
fig, axes = plt.subplots(2,1)
fig.tight_layout(pad=2.0)
axes[0].plot(data_vis['X'])
axes[0].set_title('Exports from USA')
axes[0].set_ylabel('USD Billions')
axes[0].set_xticks(range(2000,2025))
axes[0].set_xticklabels(['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011',
                   '2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2023','2024'])
axes[0].legend(data_vis['X'].columns)

axes[1].plot(data_vis['M'])
axes[1].set_title('Imports from USA')
axes[1].set_ylabel('USD Billions')
axes[1].set_xticks(range(2000,2025))
axes[1].set_xticklabels(['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011',
                   '2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2023','2024'])
axes[1].legend(data_vis['M'].columns)

for date_C in Crisis.index[Crisis==1]:
    axes[0].axvline(x=date_C, color='gray', alpha=0.8, label='Crisis', linestyle='--')
    axes[1].axvline(x=date_C, color='gray', alpha=0.8, label='Crisis', linestyle='--')

for date_T in [2018,2019,2020,2021]:
    axes[0].axvline(x=date_T, color='red', alpha=0.5, label='Trump')
    axes[1].axvline(x=date_T, color='red', alpha=0.5, label='Trump')

plt.show()

#Dummies for Fixed effects creation
data_all['Trump'] = np.where((data_all['date']>=2018) & (data_all['date']<=2021),1,0)
data_all['Pairs'] = np.where(data_all['Importer'] != 'USA','USA' + '_' + data_all['Importer'],'USA' + '_' + data_all['Exporter'])
data_all = pd.get_dummies(data_all, columns=['Exporter', 'Importer', 'Pairs'], drop_first=True)

#Gravitational model estimation OLS
X =data_all.drop(columns=['date','Trade_flow'])
X = sm.add_constant(X)
X = X.astype(float)
Y = data_all['Trade_flow']
ols = sm.OLS(Y,X).fit()
print(ols.summary(),'\n')

#Gravitational model estimation PPML
ppml_formula = """Trade_flow ~ Crisis + Exporter_GDP + Importer_GDP + Trump 
+ Exporter_China + Exporter_Germany + Exporter_Italy + Exporter_Japan 
+ Exporter_Mexico + Exporter_Canada + Exporter_USA
+ Importer_China + Importer_Germany + Importer_Italy + Importer_Japan 
+ Importer_Mexico + Importer_Canada + Importer_USA 
+ Pairs_USA_China + Pairs_USA_Germany + Pairs_USA_Italy + Pairs_USA_Japan
+ Pairs_USA_Mexico + Pairs_USA_Canada"""
ppml_model = smf.glm(formula=ppml_formula, data=data_all, family=sm.families.Poisson()).fit()
print(ppml_model.summary(),'\n')
