#Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Data from distributions as ictionary of lists
sample_size = 1000
iterations = 500
np.random.seed(1)
distribution_means = {'Normal':[],
                      'Uniform':[],
                      'Gamma':[],
                      'Poisson':[],
                      'Exponential':[],
                      'Beta':[],
                      'Weibull':[],
                      'LogNormal':[]}
for i in range(iterations):
    Normal = np.random.normal(loc=0, scale=1, size=sample_size)
    Uniform = np.random.uniform(low=0, high=15, size=sample_size)
    Gamma = np.random.gamma(shape=2, scale=2, size=sample_size)
    Poisson = np.random.poisson(lam=5, size=sample_size)
    Exponential = np.random.exponential(scale=1, size=sample_size)
    Beta = np.random.beta(a=2, b=4, size=sample_size)
    Weibull = np.random.weibull(a=2, size=sample_size)
    LogNormal = np.random.lognormal(mean=0, sigma=1, size=sample_size)
    distribution_means['Normal'].append(np.mean(Normal))
    distribution_means['Uniform'].append(np.mean(Uniform))
    distribution_means['Gamma'].append(np.mean(Gamma))
    distribution_means['Poisson'].append(np.mean(Poisson))
    distribution_means['Exponential'].append(np.mean(Exponential))
    distribution_means['Beta'].append(np.mean(Beta))
    distribution_means['Weibull'].append(np.mean(Weibull))
    distribution_means['LogNormal'].append(np.mean(LogNormal))
    if i == 199:
        fig_I, axes_I = plt.subplots(2,4)
        fig_I.tight_layout(pad=3)
        fig_I.suptitle('Original Distributions')
        axes_I = axes_I.reshape(-1)
        for axes, distr in enumerate(distribution_means.keys()):
            axes_I[axes].hist(locals()[distr], linewidth=1, edgecolor='black')
            axes_I[axes].set_title(f'Histogram of the {distr} distribution')
            axes_I[axes].set_ylabel('Frequency')
            axes_I[axes].set_xlabel('Values')

distribution_means = {k: pd.Series(lista) for k, lista in distribution_means.items()} #Optional to transform lists into Series
    
fig_II, axes_II = plt.subplots(2,4)
fig_II.tight_layout(pad=3)
fig_II.suptitle('Distributions of the Means of each Original Distribution')
axes_II = axes_II.reshape(-1)
for ax, (distr, mean) in enumerate(distribution_means.items()):
    axes_II[ax].hist(mean, linewidth=1, edgecolor='black', color='green')
    axes_II[ax].set_title(f'Histogram of means of {iterations} {distr} distributions')
    axes_II[ax].set_ylabel('Frequency')
    axes_II[ax].set_xlabel('Means Values')
plt.show()
