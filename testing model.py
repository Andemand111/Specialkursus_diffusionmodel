from model import Model
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people, load_digits
from data import Data


data, _ = load_digits(return_X_y=True, shuffle=True)
data = data / 15
h,b,c = 8,8,1

# data, _ = fetch_lfw_people(return_X_y = True, shuffle=True, color=True)
# h, b, c = 62,47,3

dataset = Data(data)

args = [h*b*c, 1000, 0.0001, 0.002]
model = Model(*args)
model.load_model("diffusion_model_digits")

#%%
""" Sample method 1 """
x_t = torch.randn((1, model.img_size))
for t in tqdm(reversed(range(model.time_steps))):
    t = torch.tensor(t)
    time_encoding = model.time_encoding(t.view((1,)))
    x_0 = model(x_t, time_encoding)

    if t > 0:
        x_t, _ = model.make_noisy_image(x_0, t)
        
plt.imshow(x_0.detach().view(h,b,c))

#%%

""" Sample method 2 """

x_t = torch.zeros((1, model.img_size))
for t in tqdm(reversed(range(model.time_steps))):
    time_encoding = model.time_encoding(torch.tensor(t).view((1,)))
    noise = torch.randn_like(x_t)
    noise_scale = 1.0 / ((t + 1) ** 0.5)
    x_t = model(x_t + noise_scale * noise, time_encoding)

plt.imshow(x_t.detach().view(h,b,c))

#%%
for j in [0, 100, 200, 300, 400]:
    ts = [100, 300, 500, 700, 900]
    fig,axs = plt.subplots(2,len(ts))
    for i, t in enumerate(ts):
        t = torch.tensor(t).view((1,))
        enc = model.time_encoding(t)
        noisy, _ = model.make_noisy_image(dataset[j].view((1,-1)), t)
        noisy = torch.clamp(noisy, 0, 1)
        reconstructed = model(noisy.view(1,-1), enc).detach()
        axs[0,i].imshow(noisy.view(h,b,c))
        axs[1,i].imshow(reconstructed.view(h,b,c))
        axs[0,i].set_xticks([])
        axs[1,i].set_xticks([])
        axs[0,i].set_yticks([])
        axs[1,i].set_yticks([])
    plt.show()
