#%%
# Import necessary libraries.
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import torch
#%%
# Download and load the training data.
trainset = datasets.MNIST('MNIST_data/', download=True)
#%%
# Make smaller dataset for testing.
trainset = trainset.data[:1000]


#%%
# Define the noising schedule.
time_steps = 1000
beta_start = 0.001
beta_end = 0.02

beta = torch.linspace(beta_start, beta_end, time_steps)
#%%
# Define prior p(x_T) distribution parameters.
p_mu = torch.zeros(time_steps)
p_std = torch.ones(time_steps)
p_T = torch.distributions.normal.Normal(p_mu, p_std)
#%%
# Define posterior q(x_T|x_0) distribution parameters.
alpha = 1 - beta
alpha_hat = torch.cumprod(alpha, dim=0)
sqrt_alpha = torch.sqrt(alpha)
sqrt_alpha_hat = torch.sqrt(alpha_hat)
one_minus_alpha_hat = 1 - alpha_hat
sqrt_one_minus_alpha_hat = torch.sqrt(one_minus_alpha_hat)


q_mu = sqrt_alpha_hat
q_std = sqrt_one_minus_alpha_hat
q_xT_given_q_x0 = torch.distributions.normal.Normal(q_mu, q_std)
#%%

KL = torch.distributions.kl_divergence(q_xT_given_q_x0, p_T)
KL = KL.detach().numpy()



# # Function to calculate KL divergence between two Gaussian distributions.
# def KL_divergence(mu_1, std_1, mu_2, std_2):
#     KL = np.log(std_2 / std_1) + (std_1**2 + (mu_1 - mu_2)**2) / (2 * std_2**2) - 0.5
#     return KL


#print(KL_divergence(q_mu, q_std, p_mu, p_std))

# %%
# Plot the KL divergence.
plt.plot(beta, KL.squeeze(), label="KL divergence")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$D_{KL}(q(x_T|x_0)||p(x_T))$")
plt.xlim(beta_start, beta_end)

# %%
print(q_mu[-5:],q_std[-5:],p_mu[-5:],p_std[-5:])

# %%

def make_noisy_image(x, t):
        eps = torch.randn(x.shape)
        x_t = sqrt_alpha_hat[t] * x + sqrt_one_minus_alpha_hat[t] * eps
        
        return x_t, eps


def make_noisy_distribution(t):
    q_mu = sqrt_alpha_hat[t] * trainset.data[0].view(-1,1).squeeze()
    q_std = sqrt_one_minus_alpha_hat[t]
    q_xT_given_q_x0 = torch.distributions.multivariate_normal.MultivariateNormal(q_mu, torch.eye(784)*q_std**2)
    return q_xT_given_q_x0


#%%

p_T = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(784), torch.eye(784))

KL_list = np.zeros(1000)
for i in range(1000):
    q_xT_given_q_x0 = make_noisy_distribution(i)
    KL_list[i] = torch.distributions.kl_divergence(q_xT_given_q_x0, p_T).detach().numpy()



#%%
# Plot the KL divergence between the noisy distribution and the prior. With beta range [0.001, 0.02].
plt.plot(KL_list)
plt.title("KL divergence between noisy distribution and prior, \n with beta range [0.001, 0.02] and t = 1000")
plt.xlabel("t")
plt.ylabel("KL divergence")
plt.show()


#%%
# Plot a normal mnist image.
plt.imshow(trainset.data[0], cmap='gray')


# %%
# Make the noisified images.
x_0 = trainset.data[0].reshape(-1, 1)
noisified_images = np.array([make_noisy_image(trainset.data[0], t)[0] for t in range(time_steps)])



# %%
# make a subplot that shows the noisified images. There should be a plot for every 200th time step.
fig, axs = plt.subplots(1,6)
for i in range(6):
    axs[i].imshow(noisified_images[i*199], cmap='gray')
    axs[i].set_title(f"t = {i*199}")
    axs[i].axis('off')
plt.show()
#%%

# KL divergence with respect to noisified images.
p_mu = torch.zeros(len(trainset.data[0].view(-1, 1)))
p_std = torch.eye(len(trainset.data[0].view(-1, 1)))
p_T = torch.distributions.multivariate_normal.MultivariateNormal(p_mu, p_std)
q_xT_given_q_x0 = torch.distributions.multivariate_normal.MultivariateNormal(noisified_images[-1].view(-1,1), sqrt_one_minus_alpha_hat[-1]*torch.eye(len(noisified_images[-1].view(-1,1))))
#%%
KL_pic = torch.distributions.kl_divergence(q_xT_given_q_x0 , p_T)
KL_pic = KL_pic.detach().numpy()
# %%
# Plot the KL divergence.
plt.plot(beta, KL_pic.squeeze(), label="KL divergence")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$D_{KL}(q(x_T|x_0)||p(x_T))$")
plt.xlim(beta_start, beta_end)
# %%
