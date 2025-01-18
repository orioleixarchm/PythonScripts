import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#Defining parameters for API
api_key = 'f10516bd8938e6e62aae398d7394b76f'
fred = Fred(api_key=api_key)

#Retrieving data series from API
series_ids = {
    'GDP':'GDP',
    'Unemployment':'UNRATE',
    'Inflation':'CPIAUCSL',
    'Interest_Rate':'FEDFUNDS'
}
data = {}
for key, id in series_ids.items():
    data[key] = fred.get_series(id)

gdp = pd.Series(data['GDP']) #Quarterly data
unemployment = pd.Series(data['Unemployment']) #Monthly data
inflation = pd.Series(data['Inflation']) #Monthly data
interest = pd.Series(data['Interest_Rate']) #Daily data

#Transforming and resampling data
df = {}
for serie, nombre in zip([gdp, unemployment, inflation, interest], ['GDP', 'Unemployment', 'Inflation', 'Interest_Rate']):
    serie.index = pd.to_datetime(serie.index)
    serie = serie.resample('M').ffill()
    df[nombre] = serie
    print(f'{nombre} series:')
    print('--------------------------')
    print(f'Length of the serie (in months): {len(serie)}')
    print(f'Number of NaNs: {serie.isna().sum()}','\n')

inflation_pct = inflation.pct_change(periods=12).mul(100) #Monthly data
df = pd.DataFrame(df)
df['Inflation_Yearly'] = df['Inflation'].pct_change(periods=12).mul(100)

#Visuals from Original data
fig, ax = plt.subplots(3,1)
fig.tight_layout(pad=2)
ax[0].plot(unemployment, label='Unemployment', color='purple')
ax[0].plot(inflation_pct, label='Yearly change Inflation', color='green')
ax[0].plot(interest, label='Interest Rate', color='orange')
ax[0].set_title('Interest Rate, Inlation and Unemployment')
ax[0].set_ylabel('Value in %')
ax[0].legend()
ax[1].plot(gdp, color='blue')
ax[1].set_title('GDP')
ax[1].set_ylabel('Millions US$')

ax[2].plot(unemployment, label='Unemployment', color='purple')
ax[2].plot(inflation_pct, label='Yearly change Inflation', color='green')
ax[2].plot(interest, label='Interest Rate', color='orange')
ax[2].set_title('Interest Rate, Inlation, Unemployment and GDP')
ax[2].set_ylabel('Value in %')
ax[2].legend()
ax[2].axvline(x=unemployment.idxmax(), color='red', linestyle='--', alpha=0.7)
ax[2].text(x=unemployment.idxmax(), y=5, color='red', s='Covid Crisis', rotation=90, va='bottom', ha='right')
ax2 = ax[2].twinx()
ax2.plot(gdp, color='blue', label='GDP')
ax2.set_ylabel('Millions US$')
ax2.legend(loc='upper right')
plt.show()

#Detrending methods
##De-meaning
gdp_demean = gdp - gdp.mean()
inflation_pct_demean = inflation_pct - inflation_pct.mean()
unemployment_demean = unemployment - unemployment.mean()
interest_demean = interest - interest.mean()

fig, axes = plt.subplots(2,2)
fig.suptitle('De-meaning method')
axes[0,0].plot(gdp_demean)
axes[0,0].set_title('GDP')
axes[0,0].set_ylabel('Millions US$')
axes[0,1].plot(inflation_pct_demean)
axes[0,1].set_title('% Change Yearly Inflation')
axes[0,1].set_ylabel('% Value')
axes[1,0].plot(unemployment_demean)
axes[1,0].set_title('Unemployment')
axes[1,0].set_ylabel('% Value')
axes[1,1].plot(interest_demean)
axes[1,1].set_title('Interest Rate')
axes[1,1].set_ylabel('% Value')

##Differencing
gdp_diff = gdp.diff()
inflation_pct_diff = inflation_pct.diff()
unemployment_diff = unemployment.diff()
interest_diff = interest.diff()

fig1, axes1 = plt.subplots(2,2)
fig1.suptitle('Differencing method')
axes1[0,0].plot(gdp_diff)
axes1[0,0].set_title('GDP')
axes1[0,0].set_ylabel('Millions US$')
axes1[0,1].plot(inflation_pct_diff)
axes1[0,1].set_title('% Change Yearly Inflation')
axes1[0,1].set_ylabel('% Value')
axes1[1,0].plot(unemployment_diff)
axes1[1,0].set_title('Unemployment')
axes1[1,0].set_ylabel('% Value')
axes1[1,1].plot(interest_diff)
axes1[1,1].set_title('Interest Rate')
axes1[1,1].set_ylabel('% Value')

##Linear trend
def linear_detrend(serie):
    serie = serie.dropna()
    time = np.arange(len(serie)).reshape(-1,1)
    model = LinearRegression().fit(time, serie.values)
    trend = model.predict(time)
    return serie - trend

gdp_lin = linear_detrend(gdp)
inflation_pct_lin = linear_detrend(inflation_pct)
unemployment_lin = linear_detrend(unemployment)
interest_lin = linear_detrend(interest)

fig2, axes2 = plt.subplots(2,2)
fig2.suptitle('Linear detrend method')
axes2[0,0].plot(gdp_lin)
axes2[0,0].set_title('GDP')
axes2[0,0].set_ylabel('Millions US$')
axes2[0,1].plot(inflation_pct_lin)
axes2[0,1].set_title('% Change Yearly Inflation')
axes2[0,1].set_ylabel('% Value')
axes2[1,0].plot(unemployment_lin)
axes2[1,0].set_title('Unemployment')
axes2[1,0].set_ylabel('% Value')
axes2[1,1].plot(interest_lin)
axes2[1,1].set_title('Interest Rate')
axes2[1,1].set_ylabel('% Value')

##Polynmial detrend (Squared)
def poly_detrend(serie, degree=2):
    serie=serie.dropna()
    time=np.arange(len(serie))
    coeff=np.polyfit(time, serie, degree)
    trend=np.polyval(coeff, time)
    return serie - trend

gdp_poly = poly_detrend(gdp, degree=2)
inflation_pct_poly = poly_detrend(inflation_pct, degree=2)
unemployment_poly = poly_detrend(unemployment, degree=2)
interest_poly = poly_detrend(interest, degree=2)

fig3A, axes3A = plt.subplots(2,2)
fig3A.suptitle('Squared detrend method')
axes3A[0,0].plot(gdp_poly)
axes3A[0,0].set_title('GDP')
axes3A[0,0].set_ylabel('Millions US$')
axes3A[0,1].plot(inflation_pct_poly)
axes3A[0,1].set_title('% Change Yearly Inflation')
axes3A[0,1].set_ylabel('% Value')
axes3A[1,0].plot(unemployment_poly)
axes3A[1,0].set_title('Unemployment')
axes3A[1,0].set_ylabel('% Value')
axes3A[1,1].plot(interest_poly)
axes3A[1,1].set_title('Interest Rate')
axes3A[1,1].set_ylabel('% Value')

##Polynomia detrend (Cubic)
gdp_poly = poly_detrend(gdp, degree=3)
inflation_pct_poly = poly_detrend(inflation_pct, degree=3)
unemployment_poly = poly_detrend(unemployment, degree=3)
interest_poly = poly_detrend(interest, degree=3)

fig3B, axes3B = plt.subplots(2,2)
fig3B.suptitle('Cubic detrend method')
axes3B[0,0].plot(gdp_poly)
axes3B[0,0].set_title('GDP')
axes3B[0,0].set_ylabel('Millions US$')
axes3B[0,1].plot(inflation_pct_poly)
axes3B[0,1].set_title('% Change Yearly Inflation')
axes3B[0,1].set_ylabel('% Value')
axes3B[1,0].plot(unemployment_poly)
axes3B[1,0].set_title('Unemployment')
axes3B[1,0].set_ylabel('% Value')
axes3B[1,1].plot(interest_poly)
axes3B[1,1].set_title('Interest Rate')
axes3B[1,1].set_ylabel('% Value')
plt.show()

##Additive decomposition
def add_decomposition(serie, freq=12, method='additive'):
    serie= serie.dropna()
    decomposition=seasonal_decompose(serie, model=method, period=freq)
    trend=decomposition.trend
    seasonality=decomposition.seasonal
    detrend=serie - trend
    return [trend, seasonality, detrend]

gdp_addc = add_decomposition(gdp)
inflation_pct_addc = add_decomposition(inflation_pct)
unemployment_addc = add_decomposition(unemployment)
interest_addc = add_decomposition(interest)

for serie, name in zip([gdp_addc, inflation_pct_addc, unemployment_addc, interest_addc], ['GDP', 'Yearly Inflation', 'Unemployment', 'Interest Rate']):
    fig4, axes4 = plt.subplots(3,1)
    fig4.tight_layout(pad=2)
    fig4.suptitle(f'{name} Seasonal and Trend Decomposition')
    axes4[0].plot(serie[0], color='blue')
    axes4[0].set_title('Series Trend')
    axes4[1].plot(serie[1], color='red')
    axes4[1].set_title('Seasonal Component')
    axes4[2].plot(serie[2], color='green')
    axes4[2].set_title('Detrended Series')
    plt.show()

##HP (Hodrick Prescott)
def hp_method(serie, lamb=1600):
    serie = serie.dropna()
    cycle, trend = hpfilter(serie, lamb)
    return [trend, cycle]

gdp_hp = hp_method(gdp, 129600)
inflation_hp = hp_method(inflation, 129600)
unemployment_hp = hp_method(unemployment, 129600)
interest_hp = hp_method(interest, 1600000)

for serie, name in zip([gdp_hp, inflation_hp, unemployment_hp, interest_hp], ['GDP', 'Inflation', 'Unemployment', 'Interest rate']):
    fig5, (ax5, ax6) = plt.subplots(2,1)
    fig5.suptitle(f'{name} Hodrick Prescott Decomposition')
    ax5.plot(serie[0], color='blue')
    ax5.set_title('Trend')
    ax6.plot(serie[1], color='red')
    ax6.set_title('Cycle')
    plt.show()

#Time series forecasting
##Stationarity and AR MU tests
for serie, nombre in zip([gdp, unemployment, interest, inflation, inflation_pct], ['GDP', 'Unemployment', 'Interest Rates', 'Inflation', 'Year Change in Inflation']):
    lag_map = {'GDP':60, 'Unemployment':100, 'Interest Rates':150, 'Inflation':100, 'Year Change in Inflation':100}
    lag = lag_map[nombre]
    serie = serie.dropna()
    test = adfuller(serie)
    print(f'Serie: {nombre}')
    print('------------------------')
    print(f'ADF Statistic is {round(test[0],5)} and  p-value is: {round(test[1],5)}.')
    i=0
    while test[1] > 0.05:
        i+=1
        test = adfuller(serie)
        print(f'Integration of order {i}')
        serie = serie.diff().dropna()
    plt.subplot(2,1,1)
    plot_acf(serie, lags=lag, ax=plt.gca(), title=f"{nombre} ACF (Autocorrelation) Moving Average term q")
    plt.subplot(2,1,2)
    plot_pacf(serie, lags=lag, ax=plt.gca(), title=f"{nombre} PACF (Partial Autocorrelation) Autoregressive term p")
    plt.show()
    print('\n')

for serie, nombre in zip([gdp, unemployment, interest, inflation, inflation_pct], ['GDP', 'Unemployment', 'Interest Rates', 'Inflation', 'Year Change in Inflation']):
    parameters_map = {'GDP':[5,2,2,4], 'Unemployment':[2,0,10,12], 'Interest Rates':[3,0,10,30], 'Inflation':[9,2,3,12], 'Year Change in Inflation':[4,0,10,12]}
    AR = parameters_map[nombre][0]
    I = parameters_map[nombre][1]
    MA = parameters_map[nombre][2]
    S = parameters_map[nombre][3]
    train = serie[:round(len(serie)*0.75)] if nombre == 'GDP' else serie[:round(len(serie)*0.8)]
    test = serie[-round(len(serie)*0.25):] if nombre == 'GDP' else serie[-round(len(serie)*0.2):] 
    arima_model = ARIMA(train,order=(AR,I,MA))
    fitted_arima = arima_model.fit()
    arima_forecast = fitted_arima.predict(start=len(train), end=len(train)+len(test)-1)

    fig6, axes7 = plt.subplots(3,1)
    fig6.tight_layout(pad=2)
    axes7[0].plot(train, color='blue', label='Train data')
    axes7[0].plot(test, color='orange', label='Test data')
    axes7[0].plot(arima_forecast, color='red', label='Forecast data')
    axes7[0].set_title(f'Train, test and ARIMA forecasted data for serie {nombre}')
    axes7[0].legend()
    axes7[1].plot(add_decomposition(serie)[0],color='green', label='trend')
    axes7[1].plot(arima_forecast, color='red', label='Forecast data')
    axes7[1].set_title(f'Trend and ARIMA forecasted data for serie {nombre}')
    axes7[1].legend()
    axes7[2].plot(test, color='orange', label='Test data')
    axes7[2].plot(arima_forecast, color='red', label='Forecast data')
    axes7[2].set_title(f'Test and ARIMA forecasted data for serie {nombre}')
    axes7[2].legend()

    fig7, axes8 = plt.subplots(3,1)
    fig7.tight_layout(pad=2)
    axes8[0].plot(train, color='blue', label='Train data')
    axes8[0].plot(test, color='orange', label='Test data')
    axes8[0].plot(es_forecast, color='red', label='Forecast data')
    axes8[0].set_title(f'Train, test and Exp. Smoothing forecasted data for serie {nombre}')
    axes8[0].legend()
    axes8[1].plot(add_decomposition(serie)[0],color='green', label='trend')
    axes8[1].plot(es_forecast, color='red', label='Forecast data')
    axes8[1].set_title(f'Trend and Exp. Smoothing forecasted data for serie {nombre}')
    axes8[1].legend()
    axes8[2].plot(test, color='orange', label='Test data')
    axes8[2].plot(es_forecast, color='red', label='Forecast data')
    axes8[2].set_title(f'Test and Exp. Smoothing forecasted data for serie {nombre}')
    axes8[2].legend()
    plt.show()
