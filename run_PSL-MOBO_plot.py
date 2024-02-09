"""
Runing the proposed Paret Set Learning (PSL) method on 15 test problems.
"""
import argparse
import numpy as np
import torch
import pickle
import cloudpickle
import os
import time
import pdb
import matplotlib.pyplot as plt
from problem import get_problem
from partitioning import sampling_vector_randomly, sampling_vector_evenly

from lhs import lhs

from pymoo.indicators.hv import HV
# from pymoo.factory import get_performance_indicator as HV

from pymoo.config import Config
Config.warnings ['not_compiled'] = False

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from mobo.surrogate_model import GaussianProcess
from mobo.transformation import StandardTransform

from model import ParetoSetModel_MLP as ParetoSetModel

import random

def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



# -----------------------------------------------------------------------------


ins_list = ['VLMOP2']

# number of initialized solutions
n_init = 10
# number of iterations, and batch size per iteration
n_iter = 20

n_sample = 10


# PSL 
# number of learning steps
n_steps = 1000 
# coefficient of LCB
coef_lcb = 0.1

# number of preference region and preference vector per region
n_region = 20
n_candidate_per_region = 5

# number of sampled candidates on the approxiamte Pareto front
n_candidate = n_region * n_candidate_per_region   
# number of optional local search
n_local = 1


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()
learning_rate = args.lr
# device
device = f'cuda:{args.gpu}' 


# -----------------------------------------------------------------------------

hv_list = {}
for test_ins in ins_list:
    set_seed(44)
    
    n_dim = 1

    start = time.time()
    
    suffix=f"_PSL-MOBO_{learning_rate}"
    suffix_dir = "plot"
    
    if not os.path.exists(f"logs_{suffix_dir}"):
        os.makedirs(f"logs_{suffix_dir}")
        
    # get problem info
    hv_all_value = np.zeros([n_iter])
    problem = get_problem(test_ins, n_dim = n_dim)
    
    n_dim = problem.n_dim
    n_obj = problem.n_obj
    bound = problem.bound
        
    if bound > 0:
        problem.ubound *= bound
        problem.lbound *= bound
    
    n_region_vec = n_obj
    
    # number of sampled preferences per step
    n_pref_update = n_region * n_region_vec
    
    # number of reference vector for testing and list of preference vectors
    n_test = n_candidate * n_obj
    
    alpha = np.ones(n_obj)
    pref_vec_test = np.random.dirichlet(alpha,n_test)
    
    print(f"Problem: {test_ins}\nN dim: {n_dim} \nN objective: {n_obj} \nLogs dir: logs_{suffix_dir}/\nRun: {suffix}")
    
    front_list, x_list, y_list = {}, {}, {}

    front_list = np.zeros((n_iter, n_test, n_obj))
    x_list = np.zeros((n_iter, n_test, n_dim))
    y_list = np.zeros((n_iter, n_test, n_obj))

    print("***********************************************")

    ref_point = problem.nadir_point
    ref_point = [1.1*x for x in ref_point]
    
    # TODO: better I/O between torch and np
    # currently, the Pareto Set Model is on torch, and the Gaussian Process Model is on np 
    
    # initialize n_init solutions 
    
    if bound == 0:
        x_init =  lhs(n_dim, n_init)
    else:
        x_init =  lhs(n_dim, n_init) * 2 * bound - bound
    
    
    y_init = problem.evaluate(torch.from_numpy(x_init).to(device))
    
    X = x_init
    Y = y_init.cpu().numpy()
    # print(X, Y)

    z = torch.zeros(n_obj).to(device)
    
    # n_iter batch selections 
    for i_iter in range(n_iter):
        print(f"Iteration:  {i_iter + 1 :03d}/{n_iter}  ||  Time: {(time.time() - start)/60:.2f} min")
        
        # intitialize the model and optimizer 
        psmodel = ParetoSetModel(n_dim, n_obj, bound)
        psmodel.to(device)
            
        # optimizer
        optimizer = torch.optim.Adam(psmodel.parameters(), lr=learning_rate)
        
        #  solution normalization
        # transformation = StandardTransform([0,1])
        # transformation.fit(X, Y)
        # X_norm, Y_norm = transformation.do(X, Y) 
        X_norm, Y_norm = X, Y
        
        
        # train GP surrogate model 
        surrogate_model = GaussianProcess(n_dim, n_obj, nu = 5)
        surrogate_model.fit(X_norm,Y_norm)
        
        z =  torch.min(torch.cat((z.reshape(1,n_obj),torch.from_numpy(Y_norm).to(device) - 0.1)), axis = 0).values.data
        
        # nondominated X, Y 
        nds = NonDominatedSorting()
        idx_nds = nds.do(Y_norm)
        
        X_nds = X_norm[idx_nds[0]]
        Y_nds = Y_norm[idx_nds[0]]

        # t_step Pareto Set Learning with Gaussian Process
        for t_step in range(n_steps):
            psmodel.train()
            
            # sample n_pref_update preferences
            alpha = np.ones(n_obj)
            pref = np.random.dirichlet(alpha, n_pref_update)
            # pref = sampling_vector_randomly(n_obj, n_region, n_region_vec)          
            pref_vec  = torch.tensor(pref).to(device).float() + 0.0001
            
            # get the current coressponding solutions
            x = psmodel(pref_vec)
            x_np = x.detach().cpu().numpy()

            # obtain the value/grad of mean/std for each obj
            mean = torch.from_numpy(surrogate_model.evaluate(x_np)['F']).to(device)
            mean_grad = torch.from_numpy(surrogate_model.evaluate(x_np, calc_gradient=True)['dF']).to(device)
            
            std = torch.from_numpy(surrogate_model.evaluate(x_np, std=True)['S']).to(device)
            std_grad = torch.from_numpy(surrogate_model.evaluate(x_np, std=True, calc_gradient=True)['dS']).to(device)
            
            # calculate the value/grad of tch decomposition with LCB
            value = mean - coef_lcb * std
            value_grad = mean_grad - coef_lcb * std_grad
            
            tch_idx = torch.argmax((1 / pref_vec) * (value - z), axis = 1)
            tch_idx_mat = [torch.arange(len(tch_idx)),tch_idx]
            tch_grad = (1 / pref_vec)[tch_idx_mat].view(n_pref_update,1) *  value_grad[tch_idx_mat] + 0.01 * torch.sum(value_grad, axis = 1) 

            tch_grad = tch_grad / torch.norm(tch_grad, dim = 1)[:, None]
            
            # gradient-based pareto set model update 
            optimizer.zero_grad()

            psmodel(pref_vec).backward(tch_grad)
            optimizer.step()  
            
        print(f"   Training completed:   Time: {(time.time() - start)/60:.2f} min")
            
        # solutions selection on the learned Pareto set
        psmodel.eval()

        # sample n_candidate preferences
        alpha = np.ones(n_obj)
        pref = np.random.dirichlet(alpha,n_candidate)
        # pref = sampling_vector_randomly(n_obj, n_region, n_candidate_per_region)
        pref  = torch.tensor(pref).to(device).float() + 0.0001

        # generate correponding solutions, get the predicted mean/std
        X_candidate = psmodel(pref).to(torch.float64)
        X_candidate_np = X_candidate.detach().cpu().numpy()
        Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']

        Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']
        Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
        
        # optional TCH-based local Exploitation 
        if n_local > 0:
            X_candidate_tch = X_candidate_np
            z_candidate = z.cpu().numpy()
            pref_np = pref.cpu().numpy()
            for j in range(n_local):
                candidate_mean =  surrogate_model.evaluate(X_candidate_tch)['F']
                candidate_mean_grad =  surrogate_model.evaluate(X_candidate_tch, calc_gradient=True)['dF']
                
                candidate_std = surrogate_model.evaluate(X_candidate_tch, std=True)['S']
                candidate_std_grad = surrogate_model.evaluate(X_candidate_tch, std=True, calc_gradient=True)['dS']
                
                candidate_value = candidate_mean - coef_lcb * candidate_std
                candidate_grad = candidate_mean_grad - coef_lcb * candidate_std_grad
                
                candidate_tch_idx = np.argmax((1 / pref_np) * (candidate_value - z_candidate), axis = 1)
                candidate_tch_idx_mat = [np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)]
                
                candidate_tch_grad = (1 / pref_np)[np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)].reshape(n_candidate,1) * candidate_grad[np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)] 
                candidate_tch_grad +=  0.01 * np.sum(candidate_grad, axis = 1) 
                
                X_candidate_tch = X_candidate_tch - 0.01 * candidate_tch_grad
                for i in range (n_dim):
                    X_candidate_tch[:,i] = np.clip(X_candidate_tch[:, i], a_min=problem.lbound[i].cpu(), a_max = problem.ubound[i].cpu())

            X_candidate_np = np.vstack([X_candidate_np, X_candidate_tch])
            
            Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']
            Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']
            
            Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std

        # greedy batch selection 
        best_subset_list = []
        Y_p = Y_nds 
        for b in range(n_sample):
            hv = HV(ref_point=np.max(np.vstack([Y_p,Y_candidate]), axis = 0))
            best_hv_value = 0
            best_subset = None
            
            for k in range(len(Y_candidate)):
                Y_subset = Y_candidate[k]
                Y_comb = np.vstack([Y_p,Y_subset])
                hv_value_subset = hv.do(Y_comb)
                if hv_value_subset > best_hv_value:
                    best_hv_value = hv_value_subset
                    best_subset = [k]
                    
            Y_p = np.vstack([Y_p,Y_candidate[best_subset].squeeze(0)])
            best_subset_list.append(best_subset)  
            
        best_subset_list = np.array(best_subset_list).T[0]
        
        # evaluate the selected n_sample solutions
        X_candidate = torch.tensor(X_candidate_np).to(device)
        X_new = X_candidate[best_subset_list]
        Y_new = problem.evaluate(X_new)

        X_new = X_new.squeeze(0)
        Y_new = Y_new.squeeze(1)
        X = np.vstack([X,X_new.detach().cpu().numpy()])
        Y = np.vstack([Y,Y_new.detach().cpu().numpy()])
        
        # check the current HV for evaluated solutions
        hv = HV(ref_point=np.array(ref_point))
        hv_value = hv(Y)
        hv_all_value[i_iter] = hv_value
        
        pref_vec  = torch.Tensor(pref_vec_test).to(device).float()

        x = psmodel(pref_vec)
        
        front_list[i_iter] = surrogate_model.evaluate(x.detach().cpu().numpy())['F']
        x_list[i_iter] = x.detach().cpu().numpy()
        y_list[i_iter] = problem.evaluate(x.to(device)).detach().cpu().numpy()

    
        print(f"   Testing completed:    Time: {(time.time() - start)/60:.2f} min")
    
        # store the final performance
        hv_list[test_ins] = hv_all_value
    
        print("***********************************************")

        
        np.save(f"logs_plot/evaluation_{test_ins}_X_{n_dim}{suffix}", X)
        np.save(f"logs_plot/evaluation_{test_ins}_Y_{n_dim}{suffix}", Y)
        
        np.save(f"logs_plot/front_{test_ins}_{n_dim}{suffix}", front_list)
        np.save(f"logs_plot/x_{test_ins}_{n_dim}{suffix}", x_list)
        np.save(f"logs_plot/y_{test_ins}_{n_dim}{suffix}", y_list)

        # %%
        x = np.linspace(-10, 10, 100)
        x = torch.from_numpy(x).unsqueeze(1)
        
        
        
        n_dim = 1
        surr = GaussianProcess(n_dim, n_obj, nu = 10)
        surr.fit(X[:-10],Y[:-10])
        
        X_cur, Y_cur = X[-10:], Y[-10:]
        X_pre, Y_pre = X[:-10], Y[:-10]
        print(X_cur.shape," ",Y_cur.shape)
        mean = surr.evaluate(x)['F']
        std = surr.evaluate(x, std=True)['S']
        real = problem.evaluate(x).numpy()
        f1 = real[:, 0]
        f2 = real[:, 1]
        mean_1 = mean[:, 0]
        std_1 = std[:, 0]
        mean_2 = mean[:, 1]
        std_2 = std[:, 1]
        
        plt.figure(figsize = (8, 8), layout="constrained")
        
        
        plt.scatter(X_pre, Y_pre[:, 0], color = "black", s=8, label="Prior solutions from previous iterations")
        plt.plot(x.flatten(), f1, color='blue', label=r"$f_1(x)$")
        plt.plot(X_cur, Y_cur[:, 0], 'or', label="Computed Pareto solutions at current iteration")
        plt.plot(x.flatten(), mean_1, '-', color='gray', label=r"$\hat{\mu}_1(x)$")
        plt.fill_between(x.flatten(), mean_1-std_1, mean_1+std_1, color='gray', alpha=0.2)
        a = 20
        plt.plot(np.array([3]*a), np.linspace(0, 1.5, a), '-', color='green')
        plt.plot(np.array([5]*a), np.linspace(0, 1.5, a), '-', color='green')
        plt.title(f"Iteration {(i_iter+1)}, number of variables {n_dim}", fontsize = 15)
        plt.xlabel("x", color = "blue", fontsize = 15)
        plt.ylim(0, 1.5)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncols=2)
        plt.savefig(f"logs_{suffix_dir}/{i_iter}_f1_VLMOP2.pdf")
        plt.close()
        
        plt.figure(figsize = (8, 8), layout="constrained")
        
        plt.scatter(X_pre, Y_pre[:, 1], color = "black", s=8, label="Prior solutions from previous iterations")
        plt.plot(x.flatten(), f2, color='blue', label=r"$f_2(x)$")
        plt.plot(X_cur, Y_cur[:, 1], 'or', label="Computed Pareto solutions at current iteration")
        plt.plot(x.flatten(), mean_2, '-', color='gray', label=r"$\hat{\mu}_2(x)$")
        plt.fill_between(x.flatten(), mean_2-std_2, mean_2+std_2, color='gray', alpha=0.2)
        a = 20
        plt.plot(np.array([3]*a), np.linspace(0, 1.5, a), '-', color='green')
        plt.plot(np.array([5]*a), np.linspace(0, 1.5, a), '-', color='green')
        plt.ylim(0, 1.5)
        plt.title(f"Iteration {(i_iter+1)}, number of variables {n_dim}", fontsize = 15)
        plt.xlabel("x", color = "blue", fontsize = 15)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncols=2)
        plt.savefig(f"logs_{suffix_dir}/{i_iter}_f2_VLMOP2.pdf")
        plt.close()
        
        
        x = np.linspace(3, 5, 100)
        x = torch.from_numpy(x).unsqueeze(1)
        truth = problem.evaluate(x).numpy() 
        
        plt.figure(figsize = (8, 8), layout="constrained")

        plt.plot(truth[:, 0], truth[:, 1], color='black', label="Truth Pareto Front")
        plt.scatter(y_list[i_iter, :, 0], y_list[i_iter, :, 1], color = "red", label= r"PF computed by Black-Box Functions $f_i(x)$")

        plt.scatter(front_list[i_iter, :, 0],front_list[i_iter, :, 1], color='green', label=r"PF computed by Surrogate funcs $\hat{f}_i(x) = \hat{\mu}_i(x)$")
        plt.xlabel(r"$f_1(x)$", color = "blue", fontsize = 15)
        plt.ylabel(r"$f_2(x)$", color = "blue", fontsize = 15)
        plt.title(f"Iteration {(i_iter+1)}, number of variables {n_dim}")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncols = 2)
        plt.savefig(f"logs_{suffix_dir}/{i_iter}_Pareto_VLMOP2.pdf")
        plt.close()
