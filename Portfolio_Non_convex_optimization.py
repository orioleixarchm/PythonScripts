#Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize

#Parameters
tickers = ['NEE', 'BEP', 'ENPH', 'FSLR', 'PLUG', 'BLDP', 'FCEL', 'SEDG', 'CSIQ']
company_names = ["NextEra_Energy","Brookfield_Renewable","Enphase_Energy",
                 "First_Solar","Plug_Power","Ballard_Power_Systems","FuelCell_Energy",
                 "SolarEdge_Technologies","Canadian_Solar"]

#Retrievening data
data = {}
for ticker, name in zip(tickers, company_names):
    data[name] = yf.Ticker(ticker=ticker).history(period='1d', start='2020-01-01')['Close']

data = pd.DataFrame(data).dropna()
returns = data.pct_change().dropna()
print(data.head(), '\n')

#Visuals
fig, (ax1, ax2) = plt.subplots(2,1)
fig.tight_layout(pad=2)

ax1.plot(data)
ax1.set_title('Stock Prices of Renewable Energy Companies')
ax1.set_ylabel('USD ($)')
ax1.legend(data.columns)

ax2.plot(returns)
ax2.set_title("Returns on Renewable Energy Companies' shares")
ax2.set_ylabel('Percentage Change (%)')
ax2.legend(returns.columns)

plt.show()

#Optimization Problem setup
def objective(W):
    exp_returns = np.sum(np.dot(W, returns.mean()))
    risk = W.T @ returns.cov() @ W
    return -(exp_returns/risk)
constraints = {'type': 'eq', 'fun': lambda W: np.sum(W) - 1}
Initial_guess = np.ones(len(tickers))/len(tickers)
bounds = [(0, 1) for _ in range(len(tickers))]
result = minimize(objective, Initial_guess, constraints=constraints, bounds=bounds, method='SLSQP')
optimal_weights = result.x

for company, weight in zip(data.columns, optimal_weights.round(5)):
    print(f'{company}: {weight if weight > 0.0005 else "Negligible Share"}')

optimal_return = np.sum(np.dot(optimal_weights, returns.mean()))
optimal_risk = np.sqrt(optimal_weights.T @ returns.cov() @ optimal_weights)

print(f"\nExpected Portfolio Return: {optimal_return:.4%}")
print(f"Portfolio Risk (Volatility): {optimal_risk:.4%}")
print(f"Sharpe Ratio: {(optimal_return / optimal_risk):.4f}")
