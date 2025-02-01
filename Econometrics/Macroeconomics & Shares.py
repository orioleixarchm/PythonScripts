#Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from fredapi import Fred
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Parameters
api_key = 'f10516bd8938e6e62aae398d7394b76f'
fred = Fred(api_key=api_key)
ticker = 'SPY'

#Retrieving data
GDP = fred.get_series('GDP', observation_start='01-01-1980')
CPI = fred.get_series('CPIAUCSL', observation_start='01-01-1980')
Unemployment = fred.get_series('UNRATE', observation_start='01-01-1980')
Interest_rate = fred.get_series('FEDFUNDS', observation_start='01-01-1980')
SPY = yf.Ticker(ticker).history(period='3mo', start='1980-01-01')['Close']

#Data manipulation
GDP.index = GDP.index.tz_localize(None)  
SPY.index = SPY.index.tz_localize(None)  
CPI.index = CPI.index.tz_localize(None)
Unemployment.index = Unemployment.index.tz_localize(None)
Interest_rate.index = Interest_rate.index.tz_localize(None)

GDP = GDP.resample('QE').mean()
GDP_change = GDP.pct_change().mul(100)
SPY = SPY.resample('QE').mean()
SPY_change = SPY.pct_change().mul(100)
CPI = CPI.resample('QE').mean()
CPI_change = CPI.pct_change(12).mul(100)
Unemployment = Unemployment.resample('QE').mean()
Unemployment_change = Unemployment.pct_change().mul(100)
Interest_rate = Interest_rate.resample('QE').mean()
Interest_rate_change = Interest_rate.pct_change().mul(100)

macro_data = pd.DataFrame({'GDP': GDP, 'SPY': SPY, 'CPI':CPI, 'Unemployment': Unemployment, 'Interest_rate': Interest_rate}).dropna()
change_data = pd.DataFrame({'GDP_change': GDP_change, 'SPY_change': SPY_change, 'CPI_change':CPI_change, 
                            'Unemployment_change': Unemployment_change, 'Interest_rate_change': Interest_rate_change}).dropna()

print(macro_data.head(), '\n')
print(change_data.head(), '\n')

#Visuals
fig, (ax1, ax2) = plt.subplots(2,1)
fig.suptitle('Macroeconomic and Stock Prices Data', fontsize=16)
ax1.plot(macro_data['GDP'], color='green', label='GDP')
ax1.set_ylabel('Billions of Dollars')
ax1.set_title('GDP and SPY Prices')
ax1.legend()
ax1B = ax1.twinx()
ax1B.plot(macro_data['SPY'], color='blue', label='SPY')
ax1B.set_ylabel('USD')
ax1B.legend(loc='upper center')

ax2.plot(change_data['GDP_change'], color='green', label='GDP % Change')
ax2.plot(change_data['SPY_change'], color='blue', label='SPY % Change')
ax2.set_title('GDP and SPY % Change')
ax2.set_ylabel('Percent points (%)')
ax2.legend()

for date_start, date_end in zip([pd.to_datetime('2008-10-01'), pd.to_datetime('2020-02-01')], [pd.to_datetime('2009-03-01'), pd.to_datetime('2020-07-01')]):
    ax1.axvspan(date_start, date_end, color='red', alpha=0.5)
    ax2.axvspan(date_start, date_end, color='red', alpha=0.5)

fig1, ax3 = plt.subplots()
sns.regplot(x='GDP_change', y='SPY_change', data=change_data, ax=ax3, color='blue')
ax3.set_title('GDP vs SPY % Change')

fig2, ax4 = plt.subplots()
sns.regplot(x=change_data['CPI_change'], y=macro_data['SPY'].iloc[1:], ax=ax4, color='red')
plt.show()

#Principal Component Decomposition
scaler = StandardScaler()
macro_scaled = scaler.fit_transform(macro_data)
pca = PCA(n_components=4)
pca_results = pca.fit_transform(macro_scaled)

components_name = macro_data.columns
explained_var = pca.explained_variance_ratio_
top_comp = []

for i, (var, component) in enumerate(zip(explained_var, pca.components_)):
    top_feature = components_name[np.argsort(abs(component))[::-1][:i]]  # Get top 4 features
    top_comp.append(top_feature)  # Store as a formatted string
    print(f"PC{i+1} ({top_comp[-1]}): Explained variance = {var*100:.2f}%")

pca_df = pd.DataFrame(pca_results, index=macro_data.index, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
pca_df['Returns'] = macro_data['SPY']

sns.pairplot(pca_df)
plt.show()