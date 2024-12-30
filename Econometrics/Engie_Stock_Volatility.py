import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
import yfinance as yf

from arch import arch_model
from pmdarima.model_selection import train_test_split
from scipy.stats import chi2, jarque_bera
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.arima.model import ARIMA

spy = yf.Ticker("ENGI.PA")
hist = spy.history(start = "2010-01-04", end = "2020-02-01")
df = pd.DataFrame(hist, columns=['Close'])
print(df.head(),'\n')

df['Return'] = np.pad(np.diff(np.log(df['Close'])) * 100, (1, 0), 'constant', constant_values=np.nan)

# Add trace plot
plt.figure(figsize=(8,5))
plt.plot(df['Return'])
plt.ylabel("Return %")
plt.title('Returns of ENGIE')
plt.show()

# Plot ACF and PACF
diff_ts = df['Return'].iloc[1:]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(diff_ts, ax=ax1, lags=10)
ax1.set_ylim(-0.5, .5) 
ax1.set_title("Autocorrelation in Returns")
plot_pacf(diff_ts, ax=ax2, lags=10)
ax2.set_ylim(-0.5, .5)  
ax2.set_xlabel("Lag")  
ax2.set_title("Partial Autocorrelation in Returns")

# Plot ACF and PACF of absolute returns
abs_returns = diff_ts.abs()
fig, (axA, axB) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(abs_returns, ax=axA, lags=10)
axA.set_ylim(-.5, .5) 
axA.set_title("Autocorrelation in Absolute Returns")
plot_pacf(abs_returns, ax=axB, lags=10)
axB.set_ylim(-.5, .5)  
axB.set_title("Partial Autocorrelation in Absolute Returns")
axB.set_xlabel("Lag")  

# Fit GARCH (1,1)
y_train, y_test = train_test_split(abs_returns, train_size= 0.8)
garch_mod = arch_model(y_train, mean = "AR",  vol='Garch', p=1, q=1, rescale=False)
res_garch = garch_mod.fit()
print(res_garch.summary(),'\n')

# Forecast the test set 
yhat = res_garch.forecast(horizon = y_test.shape[0], reindex=True)
fig, (axC, axD, axE) = plt.subplots(3, 1, figsize=(10, 8))
axC.plot(diff_ts[-y_test.shape[0]:])
axC.set_title('Test Returns')

# Plot volatility estimates for test set
axD.plot(y_test.index, np.sqrt(yhat.variance.values[-1,:]))
axD.set_title('ENGIE Volatility Prediction')

# Plot mean estimates for test set
axE.plot(y_test.index, yhat.mean.values[-1,:], label='Mean Forecasted Abs Return', color='orange')
axE.plot(y_test.index, diff_ts.iloc[-y_test.shape[0]:], label='Actual Abs Return', color='b')
axE.set_title('ENGIE Returns Prediction')
axE.legend()
plt.show()

# Conditional volatility by model fitting
res_garch.plot(annualize="D")

# Rolling Predictions
rolling_preds = []
roll_returns = []

for i in range(y_test.shape[0]):
    train = abs_returns[:-(y_test.shape[0]-i)]
    model = arch_model(train, mean='AR', p=1, q=1, rescale = False)
    model_fit = model.fit(disp='off')
    # One step ahead predictor
    pred = model_fit.forecast(horizon=1, reindex=True)
    rolling_preds.append(np.sqrt(pred.variance.values[-1,:][0]))
    roll_returns.append(pred.mean.values[-1,:][0])

rolling_preds = pd.Series(rolling_preds, index=y_test.index)
roll_returns = pd.Series(roll_returns, index=y_test.index)

# Compare n-step-ahead and one-step-ahead rolling predictions
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(diff_ts[-y_test.shape[0]:])
ax1.plot(y_test.index, np.sqrt(yhat.variance.values[-1,:]))
ax1.set_title("ENGIE Volatility N-Step Predictions")
ax1.legend(['True Daily Returns', 'Predicted Volatility'])

ax2.plot(diff_ts[-y_test.shape[0]:])
ax2.plot(y_test.index,rolling_preds)
ax2.set_title("ENGIE Volatility Rolling Predictions")
ax2.legend(['True Daily Returns', 'Predicted Rolling Volatility'])

# Plot of predicted test data rolling volatility predictions
fig,ax = plt.subplots(figsize=(10,4))
plt.plot(rolling_preds)
plt.title('ENGIE Rolling Volatility Prediction')

# Compare n-step-ahead and one-step-ahead rolling predictions
fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))
ax3.plot(diff_ts[-y_test.shape[0]:])
ax3.plot(y_test.index, yhat.mean.values[-1,:])
ax3.set_title("ENGIE Returns N-Step Predictions")
ax3.legend(['True Daily Returns', 'Predicted Returns'])

ax4.plot(diff_ts[-y_test.shape[0]:])
ax4.plot(y_test.index,roll_returns)
ax4.set_title("ENGIE Returns Rolling Predictions")
ax4.legend(['True Daily Returns', 'Predicted Rolling Returns'])

fig, (ax5, ax6, ax7) = plt.subplots(3, 1, figsize=(10, 8))
ax5.plot(y_test.index, rolling_preds, label='Predicted Rolling Volatility', color='orange')
ax5.plot(y_test.index, df['Return'].iloc[-y_test.shape[0]:].rolling(window=40).std(), label='Actual Rolling Volatility', color='blue')
ax5.set_title('Predicted vs Actual Rolling Volatility')
ax5.legend()

ax6.plot(y_test.index, rolling_preds, label='Predicted Rolling Volatility', color='orange')
ax6.plot(y_test.index, y_test.rolling(window=40).std(), label='Actual Rolling Volatility', color='blue')
ax6.set_title('Predicted vs Actual Absolute Rolling Volatility')
ax6.legend()

ax7.plot(y_test.index, yhat.variance.values[-1,:], label='Predicted Variance', color='orange')
ax7.plot(y_test.index, y_test**2, label='Actual Abs Variance', color='blue')
ax7.plot(y_test.index, rolling_preds**2, label='Predicted Rolling Variance', color='green')
ax7.set_title('Predicted vs Actual Variance')
ax7.legend()

plt.show()