#Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import t

#Parameteres
random.seed(1)
rho_vec = [0.3, 0.7]
T_vec = [5,10,15,20,35,50,65,80]
R = 1000
B = 500
Buffer = 200
alpha_ci = 0.05

#Figures and plots
fig_I, ax_I = plt.subplots(2,4)
fig_I.tight_layout()
ax_I = ax_I.flatten()
fig_II, ax_II = plt.subplots(2,4)
fig_II.tight_layout()
ax_II = ax_II.flatten()
fig_III, ax_III = plt.subplots(2,4)
fig_III.tight_layout()
ax_III = ax_III.flatten()
fig_IV, ax_IV = plt.subplots(2,4)
fig_IV.tight_layout()
ax_IV = ax_IV.flatten()

#Rho Loop
for rho in rho_vec:
    p = 0
    #Sample size Loop
    for T in T_vec:
        p += 1
        #Preacreating arrays
        rho_hat_all = np.zeros((R,1))
        rho_bc_all = np.zeros((R,1))
        ols_se_all = np.zeros((R,1))
        boot_se_all = np.zeros((R,1))
        ci_ols_store = np.zeros((R,2))
        ci_bc_store = np.zeros((R,2))
        co_ci_ols = np.zeros((R,1))
        co_ci_bc = np.zeros((R,1))

        #Degrees of freedom 
        k = 1
        df = T - k - 1
        crit = t.ppf(1-alpha_ci/2, df)

        #Montecarlo Loop
        for r in range(R):
            y_ph =  np.zeros((T+Buffer,1))
            u =  np.random.randn(T+Buffer,1)
            
            #Data generation
            for i in range(1,T+Buffer):
                y_ph[i] = rho*y_ph[i-1] + u[i]
            y_ph = y_ph[-T:]

            #OLS simulation
            y = y_ph[1:]
            x = y_ph[0:-1]
            rho_hat = np.linalg.inv(x.T @ x) @ (x.T @ y)
            u = y - x*rho_hat
            u_c = u - np.mean(u)
            sigma2_hat = (u.T @ u)/df
            ols_se = np.sqrt(sigma2_hat/(x.T @ x))
            
            ci_ols =[rho_hat - crit*ols_se, rho_hat + crit*ols_se]
            ci_ols_store[r,:] = np.array(ci_ols).squeeze()
            co_ci_ols[r] = (rho >= ci_ols[0]) & (rho <= ci_ols[1])

            #Bootstrap simulation
            rho_star = np.zeros((B,1))
            for b in range(B):
                u_star_ph = u_c[np.random.randint(0,T-1,size=(T-1,1))]
                y_star_ph = np.zeros((T,1))
                y_star_ph[0] = y[0]
                for t_ in range(1,T):
                    y_star_ph[t_] = rho_hat*y_star_ph[t_-1] + u_star_ph[t_-1]
                y_star = y_star_ph[1:]
                x_star = y_star_ph[0:-1]
                rho_star[b] = np.linalg.inv(x_star.T @ x_star) @ (x_star.T @ y_star)

            #Saving Rho_hat, Rho_bc, standard error of Rho* (Bootstrapped)
            mean_rho = np.mean(rho_star)
            bias = mean_rho - rho_hat
            rho_bc = rho_hat - bias

            boot_se = np.std(rho_star, ddof=1)
            bc_se = ols_se

            ci_bc = [rho_bc - crit*bc_se, rho_bc + crit*bc_se]
            ci_bc_store[r,:] = np.array(ci_bc).squeeze()
            co_ci_bc[r] = (rho >= ci_bc[0]) & (rho <= ci_bc[1])

            rho_hat_all[r] = rho_hat
            rho_bc_all[r] = rho_bc
            ols_se_all[r] = ols_se
            boot_se_all[r] = boot_se
        
        #Main statistics of interest
        mean_rho_hat = np.mean(rho_hat_all)
        mean_rho_bc = np.mean(rho_bc_all)
        mean_bias_rho_hat = mean_rho_hat - rho
        mean_bias_rho_bc = mean_rho_bc - rho
        mean_ols_se = np.mean(ols_se_all)
        mean_boot_se = np.mean(boot_se_all)
        se_rho_hat = np.std(rho_hat_all, ddof=1)
        coverage_ols = np.mean(co_ci_ols)
        coverage_bc = np.mean(co_ci_bc)
        diff = mean_boot_se - mean_ols_se

        #Printing results
        results = pd.DataFrame({'Statistic': ['True Rho','Mean Rho_hat', 'Mean Rho_bc', 'Mean Bias Rho_hat', 
                                              'Mean Bias Rho_bc', 'Mean OLS SE', 'Mean Boot SE', 
                                              'SE of Rho_hat', 'Coverage OLS CI', 'Coverage BC CI', 
                                              'Diff mean BC - mean OLS'],
                                'Value': [rho,mean_rho_hat, mean_rho_bc, mean_bias_rho_hat,
                                          mean_bias_rho_bc, mean_ols_se, mean_boot_se,
                                          se_rho_hat, coverage_ols, coverage_bc,
                                          diff]})
        results = results.T.reset_index(drop=True)
        results.columns = results.iloc[0]
        results = results[1:]
        print(f'Results for rho = {rho} and T = {T}')
        print(results,'\n')

        #Plotting results
        ax = ax_I if rho == 0.3 else ax_III
        ax[p-1].hist(rho_hat_all, bins=30, density=True, alpha=0.3, color=(0.0, 0.2, 0.4), label=r'$\hat{\rho}$ (OLS)')
        ax[p-1].hist(rho_bc_all, bins=30, density=True, alpha=0.3, color=(0.6, 0.2, 0.2), label=r'$\hat{\rho}^{BC}$')
        ax[p-1].axvline(x=rho, color='k', linestyle='-', label=r'True $\rho$')
        ax[p-1].axvline(x=mean_rho_hat, color='k', linestyle=':', label=r'$\hat{\rho}^{ols}$')
        ax[p-1].axvline(x=rho, color='k', linestyle='-.', label=r'$\hat{\rho}^{bc}$')
        ax[p-1].set_title(f'Histogram of OLS and Bias Corrected estimators for T = {T} and $\\rho$ = {rho}')
        ax[p-1].set_xlabel('Value')
        ax[p-1].set_ylabel('Density')
        ax[p-1].legend()

        ax = ax_II if rho == 0.3 else ax_IV
        bar_vals = [mean_ols_se, se_rho_hat, mean_boot_se]
        x = np.arange(len(bar_vals))
        ax[p-1].bar(x, bar_vals, color=['blue', 'orange', 'green'], edgecolor='black', linewidth=1.2)
        ax[p-1].set_title(f'Standard Errors and Deviation for T = {T} and $\\rho$ = {rho}')
        ax[p-1].set_xlabel('Statistic')
        ax[p-1].set_ylabel(None)
        ax[p-1].set_xticks(x)
        ax[p-1].set_xticklabels(['Mean OLS SE', 'SE of Rho_hat', 'Mean Boot SE'])

plt.show()