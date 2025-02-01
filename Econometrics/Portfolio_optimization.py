#Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import cvxpy as cp

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
weights = cp.Variable(len(tickers))
expected_return = cp.sum(cp.multiply(weights, returns.mean()))
risk = cp.quad_form(weights, returns.cov())
objective = cp.Maximize(expected_return - risk)
constraints = [cp.sum(weights) == 1, weights >= 0]

#Solving the problem
Problem = cp.Problem(objective, constraints)
Problem.solve()

print('Optimal Portfolio Weights:')
for colum, weight in zip(data.columns, weights.value.round(5)):
    print(f'{colum}: {weight if weight > 00.005 else 'Negligible Share'}')
