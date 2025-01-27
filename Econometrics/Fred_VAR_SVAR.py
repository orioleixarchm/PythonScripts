#Loading packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from statsmodels.tsa.api import VAR, SVAR
from statsmodels.tsa.stattools import adfuller

#API parameters
api_key = 'f10516bd8938e6e62aae398d7394b76f'
fred = Fred(api_key=api_key)

#Loading data
GDP = fred.get_series('GDP', observation_start='1960-01-01')
CPI = fred.get_series('CPIAUCNS', observation_start='1960-01-01').div(100)
Interest_rate = fred.get_series('FEDFUNDS', observation_start='1960-01-01')
Unemployment_rate = fred.get_series('UNRATE', observation_start='1960-01-01')
Yearly_inflation = CPI.pct_change(12).mul(100)

#Visuals
fig, axes = plt.subplots(3,1)
fig.tight_layout(pad=2)
axes[0].plot(Interest_rate, color='green', label='Interest Rate')
axes[0].plot(Unemployment_rate, color='red', label='Unemployment Rate')
axes[0].plot(Yearly_inflation, color='purple', label='Yearly Inflation Rate')
axes[0].set_title('Interest, Unemployment and Year Inflation Rates')
axes[0].set_ylabel('Percent points (%)')
axes[0].legend()
axes[1].plot(GDP, color='blue') 
axes[1].set_title('Gross Domestic Product')
axes[1].set_ylabel('Billions of Dollars')

axes[2].plot(Interest_rate, color='green', label='Interest Rate')
axes[2].plot(Unemployment_rate, color='red', label='Unemployment Rate')
axes[2].plot(Yearly_inflation, color='purple', label='Yearly Inflation Rate')
axes[2].set_ylabel('Percent points (%)')
axes[2].legend()
axes2 = axes[2].twinx()
axes2.plot(GDP, color='blue', label='GDP') 
axes2.set_ylabel('Billions of Dollars')
axes2.legend()
axes[2].set_title('GDP and Interest, Unemployment and Year Inflation Rates')
plt.show()

#Resampling data to quarterly series
GDP = GDP.resample('Q').median()
CPI_q = CPI.resample('Q').median()
Interest_rate_q = Interest_rate.resample('Q').median()
Unemployment_rate_q = Unemployment_rate.resample('Q').median()
Yearly_inflation_q = Yearly_inflation.resample('Q').median()

data_q = pd.DataFrame({'GDP': GDP,
                       'CPI': CPI_q, 
                       'Interest_rate': Interest_rate_q, 
                       'Unemployment_rate': Unemployment_rate_q, 
                       'Yearly_inflation': Yearly_inflation_q}).dropna()
print(data_q.head(15), '\n')

#Checking for stationarity
def stationarity_check(serie):
    result = adfuller(serie)
    i=0
    while result[1] > 0.05:
        i+=1
        print(f'The series {serie.name} is not stationary, integration of order {i}.')
        serie = serie.diff().dropna()
        result = adfuller(serie)
    print(f'The series {serie.name} is stationary','\n')
    return serie

data_q = data_q.apply(stationarity_check).dropna()
print(data_q.head(15), '\n')

#VAR model
var_model = VAR(data_q)
var_results = var_model.fit(4)
print(var_results.summary(), '\n')

irf = var_results.irf(10)
irf.plot(orth=True)
plt.show()

fevd = var_results.fevd(10)
print(fevd.summary())
fevd.plot()
plt.title('Forecast Error Variance Decomposition')
plt.show()

#SVAR model
A = np.array([
    [1, 0, 0, 0],
    ['E', 1, 0, 0],
    ['E', 'E', 1, 0],
    ['E', 'E', 'E', 1]
])
B = np.array([
    [1, 0, 0, 0],
    ['E', 1, 0, 0],
    ['E', 'E', 1, 0],
    ['E', 'E', 'E', 1]
])

endogenous_vars = ['GDP', 'Interest_rate', 'Unemployment_rate', 'Yearly_inflation']
exogenous_vars=["CPI"]

svar_model = SVAR(endog=data_q[endogenous_vars], svar_type='A', A=A)
svar_results = svar_model.fit(maxlags=5, solver='bfgs', trend="n")
svar_results.k_exog_user = 0
print(svar_results.summary(), '\n')

irf_svar = svar_results.irf(10)
irf_svar.plot(orth=False)
plt.show()
