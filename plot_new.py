# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
import torch
class cybenko():
    def __init__(self, n_dim = 1):
        self.n_dim = n_dim
        self.n_obj = 2
        self.nadir_point = [1, 1]
       
    def evaluate(self, x):      
        if self.n_dim == 1:
            a = x - 2 
        else:
            a = x 
        n = x.shape[1]
        
        f1 = 1 - torch.exp(-torch.sum((a - 1 / np.sqrt(n))**2, axis = 1))
        f2 = 1 - torch.exp(-torch.sum((a + 1 / np.sqrt(n))**2, axis = 1))
     
        objs = torch.stack([f1,f2]).T
        
        return objs
    

# %%
# x = np.linspace(-1, 1, 100)
# x = torch.from_numpy(x).unsqueeze(1)
# x.shape

# %%
pb = cybenko(n_dim=1)
x = np.linspace(3, 5, 100)
x = torch.from_numpy(x).unsqueeze(1)
truth = pb.evaluate(x).numpy() 

# pb = disconnect()
# z = np.load("/home/ubuntu/long.hp/MultiSample-Hypernetworks/Toy_Examples/front/toy3_truth.npy")
# plt.plot(z[:, 0], z[:, 1], color='red')



# %%
import cloudpickle

# %%
import time

# %%
n_dim = "5"
plt.plot(truth[:, 0], truth[:, 1], color='red', label="Truth Pareto Front")
X = np.load("x/EHVI_X_" + n_dim + ".npy")
Y = np.load("y/EHVI_Y_" + n_dim + ".npy")
plt.scatter(Y[:, 0], Y[:, 1])

# %%
Y[np.where(Y[:, 0] + Y[:, 1] < 2)]

# %%
X

# %%
dem = 1
for i in X:
    print(i, end=" ")
    if (dem % 5) == 0:
        print()
    dem += 1

# %%
from mobo.surrogate_model import GaussianProcess

# %%
n_dim = 1
surr = GaussianProcess(1, 2, nu = 5)
surr.fit(X,Y)

# %%
x = np.linspace(-10, 10, 100)
x = torch.from_numpy(x).unsqueeze(1)
mean = surr.evaluate(x)['F']
std = surr.evaluate(x, std=True)['S']
real = pb.evaluate(x).numpy()
f1 = real[:, 0]
f2 = real[:, 1]
mean_1 = mean[:, 0]
std_1 = std[:, 0]
mean_2 = mean[:, 1]
std_2 = std[:, 1]

# %%
# truth = pb.evaluate(x).numpy()

# %%
plt.plot(X, Y[:, 0], 'or', label="Generated Pareto Optimal Sols")
plt.plot(x.flatten(), f1, color='blue', label=r"$f_1(x)$")
plt.plot(x.flatten(), mean_1, '-', color='gray', label=r"$\hat{\mu}_1(x)$")
plt.fill_between(x.flatten(), mean_1-std_1, mean_1+std_1, color='gray', alpha=0.2)
a = 20
plt.plot(np.array([3]*a), np.linspace(0, 2.5, a), '-', color='green')
plt.plot(np.array([5]*a), np.linspace(0, 2.5, a), '-', color='green')
plt.title(f"Number of expensive evaluations {i}, number of variables {n_dim}")
plt.xlabel("x")
plt.legend()
# plt.savefig("figures_uncer/f1_" + i +".png" )
plt.show()

# %%


# %%


# %%


# %%


# %%
standard_bounds = torch.zeros(2, 1)
standard_bounds[1] = 1

# %%
standard_bounds

# %%
# i_iter = [1, 30, 60, 90, 120, 150, 180, 200] #, 'last']
i_iter = [i for i in range(1, 33, 1)]
i_iter = [str(i) for i in i_iter]
n_dim = "1"
n_sample = "5"
# name = 'ascent_cybenko_GIBO'
name = 'cybenko'
for i in i_iter:
    real_func = np.load("y/" + name + "_" + n_dim + "_" + n_sample + "_" + i + ".npy")
    # real_func = np.load("y/" + name + "_" + n_dim + "_" + i + ".npy")
    surrogate_func = np.load("fronts/" + name + "_" + n_dim + "_" + n_sample + "_" + i + ".npy")
    # with open("transformations/" + name + "_" + n_dim + "_" + i + ".pkl", 'rb') as f:
    #     transformation = cloudpickle.load(f)[0]
    # surrogate_func = transformation.undo(surrogate_func)
    plt.plot(truth[:, 0], truth[:, 1], color='red', label="Truth Pareto Front")
    plt.scatter(real_func[:, 0], real_func[:, 1], label= r"PF computed by Black-Box Functions $f_i(x)$")
    plt.scatter(surrogate_func[:, 0], surrogate_func[:, 1], color='green', label=r"PF computed by Surrogate funcs $\hat{f}_i(x) = \hat{\mu}_i(x)$")
    plt.xlabel("Loss 1")
    plt.ylabel("Loss 2")
    plt.title(f"Number of expensive evaluations {i}, number of variables {n_dim}")
    # plt.plot(z[:, 0], z[:, 1], color='red', label='Truth PF')
    plt.legend()
    plt.show()

# %%


# %%


# %%


# %%


# %%
n_dim = "1"
# n_sample = "10"
# name = 'descent_cybenko_GIBO'
name = 'cybenko_GIBO'
i = '100'
real_func = np.load("y/" + name + "_" + n_dim + "_" + n_sample + "_" + i + ".npy")
# surrogate_func = np.load("fronts/" + name + "_" + n_dim + "_" + n_sample + "_" + i + ".npy")


# %%
plt.plot(truth[:, 0], truth[:, 1], color='red', label="Truth Pareto Front")
plt.scatter(real_func[:, 0], real_func[:, 1], label= r"PF computed by Black-Box Functions $f_i(x)$")
# plt.scatter(surrogate_func[:, 0], surrogate_func[:, 1], color='green', label=r"PF computed by Surrogate funcs $\hat{f}_i(x) = \hat{\mu}_i(x)$")
plt.xlabel("Loss 1")
plt.ylabel("Loss 2")
plt.title(f"Number of expensive evaluations {i}, number of variables {n_dim}")
plt.legend()

# %%



