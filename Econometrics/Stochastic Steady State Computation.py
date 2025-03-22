import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# Initialize parameters
par = {
    'alpha': 0.33,
    'beta': 0.99,
    'sigma': 2,
    'delta': 0.025,
    'Ass': 1,
    'N': 20,
    'Tol': 1e-2,
    'maxIt': int(1e5)
}

# Productivity states and transition matrix
A = np.array([0.9, 1, 1.1])
p = 0.8
P = np.array([[p, (1 - p) / 2, (1 - p) / 2],
              [(1 - p) / 2, p, (1 - p) / 2],
              [(1 - p) / 2, (1 - p) / 2, p]])

par['n'] = len(A)

# Calculate steady-state capital and grid
par['kss'] = ((par['alpha'] * par['Ass']) / (par['beta'] - (1 - par['delta']))) ** (1 / (1 - par['alpha']))
par['kmin'] = par['kss'] * 0.8
par['kmax'] = par['kss'] * 1.2
par['kgrid'] = np.linspace(par['kmin'], par['kmax'], par['N'])

# Initialize value functions and capital policy
V0 = np.zeros((par['N'], par['n']))
V1 = np.zeros((par['N'], par['n']))
kfut = np.tile(par['kgrid'].reshape(-1, 1), (1, par['n'])) #repeats the initial matrix or array n times
errV = 10
its = 0

# Define the value function
def NegValFun2(k1, k0, par, j, A, V0, P):
    A0 = A[j]  # current period productivity
    c = (1 - par['delta']) * k0 + A0 * k0 ** par['alpha'] - k1  # consumption calculation
    v0 = interp1d(par['kgrid'], V0, axis=0)(k1)
    v1 = -(c ** (1 - par['sigma']) - 1) / (1 - par['sigma']) + par['beta'] * (P[j, :] @ v0.T) #@ equivalent to np.dot() matrix mult, T for transpose
    return v1  # Negative for minimization

# Value function iteration
while errV > par['Tol'] and its < par['maxIt']:
    for i in range(par['N']):
        k0 = par['kgrid'][i]
        for j in range(par['n']):
            k_high = (1 - par['delta']) * k0 + A[j] * k0 ** par['alpha'] - 0.01
            result = minimize_scalar(NegValFun2, bounds=(par['kmin'], min(par['kmax'], k_high)),
                                     args=(k0, par, j, A, V0, P), method='bounded')
            k1 = result.x
            V1[i, j] = NegValFun2(k1, k0, par, j, A, V0, P)
            kfut[i, j] = k1

    errV = np.max((V1 - V0) ** 2)
    V0 = np.copy(V1)
    its += 1
    print(f"Iteration: {its}, Error: {errV}")


# Plotting the solution
# Plot the value function
plt.figure(11)
plt.plot(100 * (par['kgrid'] - par['kss']) / par['kss'], -V1)
plt.xlabel('k in % deviation from steady state')
plt.ylabel('value function')
plt.legend(['low A', 'steady state A', 'high A'], loc='best')
plt.title('k in % deviation from steady state and value Function')
plt.show()

# Plot the policy function for k_{t+1}
plt.figure(12)
plt.plot(100 * (par['kgrid'] - par['kss']) / par['kss'], 100 * (kfut - par['kss']) / par['kss'])
plt.xlabel('k in % deviation from steady state')
plt.ylabel(r'choice of $k_{t+1}$ (in % deviation from steady state)')
plt.legend(['low A', 'steady state A', 'high A'], loc='best')
plt.title(r'k and choice of $k_{t+1}$ in % deviation from steady state')
plt.show()

# Plot the change in capital
plt.figure(13)
plt.plot(100 * (par['kgrid'] - par['kss']) / par['kss'], 100 * (kfut - par['kgrid'].reshape(-1, 1)) / par['kss'])
plt.xlabel('k in % deviation from steady state')
plt.ylabel('change in k as % of steady state')
plt.legend(['low A', 'steady state A', 'high A'], loc='best')
plt.title(r'k and change in k in % deviation from steady state')
plt.axhline(0, color='k', linestyle='--')
plt.show()

# Plot the consumption deviation
#ct = np.kron(A, par['kgrid'] ** par['alpha']) - kfut + (1 - par['delta']) * par['kgrid'].reshape(-1, 1)
#css = par['Ass'] * par['kss'] ** par['alpha'] - par['delta'] * par['kss']
ct = (par['kgrid'].reshape(-1, 1) ** par['alpha'] @ A.reshape(1, -1)) - kfut + (1 - par['delta']) * par['kgrid'].reshape(-1, 1)
css = par['Ass'] * par['kss'] ** par['alpha'] - par['delta'] * par['kss']

fig, ax = plt.subplots()
ax.plot(100 * (par['kgrid'] - par['kss']) / par['kss'], 100 * (ct - css) / css)
axin = ax.inset_axes([0.6,.1,.3,.3]) #first position in the X axis, second, in the y axis, third, how wide, and then how tall 
axin.plot(100 * (par['kgrid'] - par['kss']) / par['kss'], 100 * (ct - css) / css)
axin.set_xlim(15,20)
axin.set_ylim(66,76)
ax.indicate_inset_zoom(axin)
axin1 = ax.inset_axes([0.2,.1,.3,.3]) #first position in the X axis, second, in the y axis, third, how wide, and then how tall 
axin1.plot(100 * (par['kgrid'] - par['kss']) / par['kss'], 100 * (ct - css) / css)
axin1.set_xlim(-17,-12)
axin1.set_ylim(23,33)
ax.indicate_inset_zoom(axin1)
plt.xlabel('k in % deviation from steady state')
plt.ylabel('consumption in % deviation from steady state')
plt.legend(['low A', 'steady state A', 'high A'], loc='best')
plt.title(r'k and c in % deviation from steady state')
plt.axhline(0, color='k', linestyle='--')
plt.show()