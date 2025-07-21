#Importing Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import weibull_min

#Parameters
np.random.seed(1)
Lambda = 20
Size_K = 3
Sample_Size = 100000
Budget = 100000

#Weibul (failure) and Log-Normal (Costs) distributions
T = np.random.weibull(a=Size_K,size=Sample_Size)*Lambda
lognormal_costs = np.random.lognormal(mean=11, sigma=0.8,size=Sample_Size)
fig_I, axes_I = plt.subplots(2,1)
fig_I.tight_layout()
sns.histplot(T, kde=True, ax=axes_I[0], edgecolor='black',linewidth=0.75)
axes_I[0].set_title(f'Weibul Distribution with K={Size_K} and Lambda={Lambda}.')
sns.histplot(lognormal_costs, kde=True, ax=axes_I[1], color='red', edgecolor='black',linewidth=0.75, bins=45)
axes_I[1].set_title(f'Log Normal Distribution with mean={11} and sigma={0.4}.')

#Prob of failure based on threshholds
min_years = [5,10,15,20,25,30]
fig_II, axes_II = plt.subplots(2,3)
fig_II.tight_layout()
axes_II = axes_II.flatten()
fig_III, axes_III = plt.subplots(2,3)
fig_III.tight_layout()
axes_III = axes_III.flatten()
for thr, axA, axB in zip(min_years,axes_II,axes_III):
    prob = np.mean(T <= thr)
    prob_exact = weibull_min.cdf(thr, c=Size_K, scale=Lambda)
    costs = np.zeros_like(T)
    costs[T <= thr] = lognormal_costs[T <= thr]
    costs = np.zeros_like(T)
    costs[T <= thr] = lognormal_costs[T <= thr]
    expected_cost = np.mean(costs)
    prob_over_budget = np.mean(costs > Budget)
    print(f"For a min threshold of {thr} years the probability is {round(prob*100,2)}%; analytically it is {round(prob_exact*100,2)} (difference of {round(prob-prob_exact,4)})")
    print(f"For a budget of {Budget}€, the expected costs are {expected_cost}€, the probability of exceeding is {round(prob_over_budget*100,2)}%.")
    sns.histplot(T < thr, kde=True, ax=axA, edgecolor='black',linewidth=0.75)
    axA.axvline(prob,color='red')
    axA.set_ylabel(None)
    axA.set_title(f"Probability of failure with a threshhold of {thr} years")
    sns.histplot(costs > Budget, kde=True, ax=axB, edgecolor='black',linewidth=0.75)
    axB.axvline(prob_over_budget,color='red')
    axB.set_ylabel(None)
    axB.set_title(f"Probability of exceeding budget with a threshhold of {thr} years")
plt.show()


