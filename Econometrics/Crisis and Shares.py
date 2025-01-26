#Loading Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import statsmodels.api as sm
from scipy.stats import ttest_ind
from fredapi import Fred

#Parameters
api_key = 'f10516bd8938e6e62aae398d7394b76f'
fred = Fred(api_key=api_key)
tickers = ['JPM', 'MSFT', 'WMT', 'XOM', 'PFE']  

#Retrieving Data
GDP = fred.get_series('GDP', observation_start='01-01-1980')
Unemployment = fred.get_series('UNRATE', observation_start='01-01-1980')
Interest_Rate = fred.get_series('FEDFUNDS', observation_start='01-01-1980')
Crisis = fred.get_series('USRECQ', observation_start='01-01-1980')
Close_prices = {}
for ticker, nombre in zip(tickers, ['JPMorgan (banking)','Microsoft (Tech)', 'Walmart (Retail)', 'ExxonMovil (Energy)', 'Pfizer (Pharma)']):
    Close_prices[nombre] = yf.Ticker(ticker).history(period='1d', start='1980-01-01')['Close']
Close_prices = pd.DataFrame(Close_prices)


#Resampling and manipulations
GDP_m = GDP.resample('ME').ffill()
Crisis_m = Crisis.resample('ME').ffill()
Unemployment_m = Unemployment.resample('ME').median()
Interest_Rate_m = Interest_Rate.resample('ME').median()
Macro_data = pd.DataFrame({'GDP':GDP_m, 'Unemployment':Unemployment_m, 'Interest_Rate':Interest_Rate_m, 'Crisis':Crisis_m})
print(Macro_data.head(), '\n')
print(Close_prices.head(), '\n')

#Visuals
fig, axes = plt.subplots(2,1)
fig.tight_layout(pad=2)
axes[0].plot(Macro_data['Interest_Rate'], color='green', label='Interest Rate')
axes[0].plot(Macro_data['Unemployment'], color='orange', label='Unemployment')
ax1 = axes[0].twinx()
ax1.plot(Macro_data['GDP'], color='blue', label='GDP')
ax1.set_ylabel('Billions of Dollars')
axes[0].set_title('Interest Rate, Unemployment and GDP')
axes[0].set_ylabel('Percent points (%)')
ax1.legend()
axes[0].legend(loc='upper left')

axes[1].plot(Close_prices)
axes[1].set_title('Stock Prices of Main Companies per Sector')
axes[1].set_ylabel('USD')
axes[1].legend(Close_prices.columns, loc='upper left')
for date in Macro_data[Macro_data['Crisis']==1].index:
    axes[0].axvline(x=date, color='gray', alpha=0.5)
    axes[1].axvline(x=date, color='gray', alpha=0.5)
plt.show()

Close_prices['Mean Close price'] = Close_prices.mean(axis=1)
Close_prices_mean_m = Close_prices['Mean Close price'].resample('ME').mean().reset_index(drop=True)
Macro_data_index = Macro_data.index
Macro_data.reset_index(drop=True, inplace=True)
Macro_data['Mean Close price'] = Close_prices_mean_m
Macro_data.index = Macro_data_index
print(Macro_data)

#Impat on shares and macro variables during crisis
crisis = Macro_data[Macro_data['Crisis']==1]
no_crisis = Macro_data[Macro_data['Crisis']==0]

for col in ['GDP', 'Interest_Rate', 'Unemployment', 'Mean Close price']:
    t_test, p_value = ttest_ind(crisis[col], no_crisis[col], nan_policy='omit')
    print(f'T-test for variable {col}: (If means significantly no crisis or in crisis)')
    print(f'T-test: {t_test}, p-value {p_value}', '\n')


for variable_y in ['GDP', 'Interest_Rate', 'Unemployment', 'Mean Close price']:
    Macro_data_I = Macro_data.dropna()
    print(f'{variable_y} during crisis:')
    print('---------------------------------')
    X = Macro_data_I['Crisis']
    X = sm.add_constant(X)
    Y = Macro_data_I[variable_y]
    model = sm.OLS(Y,X).fit()
    print(model.summary(), '\n')

#Correlations between macro variables:
Macro_data_corr = Macro_data.corr()
sns.heatmap(Macro_data_corr, annot=True)
plt.title('Correlation Matrix')
plt.xticks(rotation=0)
plt.show()