#%%
import numpy as np
import matplotlib.pyplot as plt
import torch

#%%
beta_start = 0.0001
beta_end = 0.02
time_steps = 1000
T = np.linspace(1, 1000, 1000)
beta = torch.linspace(beta_start, beta_end, time_steps)
alpha = 1 - beta
alpha_hat = torch.cumprod(alpha, dim=0)
sqrt_alpha = torch.sqrt(alpha)
sqrt_alpha_hat = torch.sqrt(alpha_hat)
one_minus_alpha_hat = 1 - alpha_hat
sqrt_one_minus_alpha_hat = torch.sqrt(one_minus_alpha_hat)


#%%
# make array to save C values in
C = np.zeros(time_steps)

# calculate C values
for t in range(1, time_steps):
    C[t] = (1 / 2*((1-alpha[t]) * (1-alpha_hat[t-1])) / (1-alpha_hat[t])) * \
    (alpha_hat[t-1] * (1-alpha[t])**2 / (1-alpha_hat[t])**2)

#%%
 # plot C values
plt.title('Constant C as a function of time')
plt.ylabel('C')
plt.xlabel('t')
# remove first entry from C and T to avoid division by zero
C = C[1:]
T = T[1:]
plt.plot(T, C, 'o', color='teal')
# %%
# checking values
print(f" first 5 values of alpha, alpha_hat, sqrt_alpha, \n sqrt_alpha_hat, one_minus_alpha_hat, sqrt_one_minus_alpha_hat:")

for i in range(5):
    print(alpha[i], alpha_hat[i], sqrt_alpha[i], sqrt_alpha_hat[i], one_minus_alpha_hat[i], sqrt_one_minus_alpha_hat[i])

#%%
# checking C values
print(f"first and last three C values: {C[0:3]} ... {C[-3:]}")
# %%
