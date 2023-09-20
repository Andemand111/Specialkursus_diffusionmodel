#%%
from model import Model
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from data import dataset
import numpy as np
#%%
args = [dataset.size, 1000, 0.0001, 0.01]
model = Model(*args)
model.load_model("diffusion_model")

#%%
""" Sample method 1 """
x_t = torch.randn((1, model.img_size))
for t in tqdm(reversed(range(model.time_steps))):
    t = torch.tensor(t)
    time_encoding = model.time_encoding(t.view((1,)))
    x_0 = model(x_t, time_encoding)

    if t > 0:
        x_t, _ = model.make_noisy_image(x_0, t)
        
plt.imshow(x_0.detach().view(*dataset.dimensions))

#%%

""" Sample method 2 """


x_t = torch.zeros((1, model.img_size))
for t in tqdm(reversed(range(model.time_steps))):
    time_encoding = model.time_encoding(torch.tensor(t).view((1,)))
    noise = torch.randn_like(x_t)
    noise_scale = 1.0 / ((t + 1) ** 0.5)
    x_t = model(x_t + noise_scale * noise, time_encoding)

plt.imshow(x_t.detach().view(*dataset.dimensions))

#%%
"""sample method 3"""
""" 
ligesom den i pdf'en, men den giver nogenlunde
samme resultater som sample method 1
"""

x_t = torch.randn((1, model.img_size))
for t in tqdm(reversed(range(model.time_steps))):
    t_enc = torch.tensor(t).view((1,))
    time_encoding = model.time_encoding(t_enc)
    x_0 = model(x_t, time_encoding)
    eps = (x_t - model.sqrt_alpha_hat[t] * x_0) / model.sqrt_one_minus_alpha_hat[t]
    
    k1 = 1 / model.sqrt_alpha[t]
    k2 = (1 - model.alpha[t]) / model.sqrt_one_minus_alpha_hat[t]
    x_t = k1 * (x_t - k2 * eps)
    if t > 0:
        sigma_sq = (1 - model.alpha[t]) * (1 - model.alpha_hat[t - 1]) / (1 - model.alpha_hat[t])
        z = torch.randn_like(eps)
        x_t += torch.sqrt(sigma_sq) * z
    
plt.imshow(x_t.detach().view(*dataset.dimensions))


#%%
""" Reconstructions given increasingly noisy images"""

for j in np.random.randint(len(dataset), size=5):
    ts = [100, 300, 500, 700, 900]
    fig,axs = plt.subplots(2,len(ts))
    for i, t in enumerate(ts):
        t = torch.tensor(t).view((1,))
        enc = model.time_encoding(t)
        noisy, _ = model.make_noisy_image(dataset[j].view((1,-1)), t)
        noisy = torch.clamp(noisy, 0, 1)
        reconstructed = model(noisy.view(1,-1), enc).detach()
        axs[0,i].imshow(noisy.view(*dataset.dimensions))
        axs[1,i].imshow(reconstructed.view(*dataset.dimensions))
        axs[0,i].set_xticks([])
        axs[1,i].set_xticks([])
        axs[0,i].set_yticks([])
        axs[1,i].set_yticks([])
        
    axs[0,2].set_title("Increasingly noisy images")
    axs[1,2].set_title("Reconstructions of above images")
    plt.show()
