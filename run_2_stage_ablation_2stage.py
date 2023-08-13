import argparse
import numpy as np
import torch
import torch.nn as nn
import pickle
import cloudpickle
import os
import time
import pdb
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

from model import ParetoSetModel

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
ins_list = ['DTLZ2']

# number of initialized solutions
n_init = 20 


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

# device
device = 'cuda:0'

parser = argparse.ArgumentParser()
parser.add_argument("--n_dim", type=int, default=15)
args = parser.parse_args()

# the dimension of optimized variables
n_dim = args.n_dim



max_evaluation = 320

# -----------------------------------------------------------------------------

hv_list = {}
for evaluation in range(30, 221, 10):
    for test_ins in ins_list:
        
        set_seed(4040)
        
        start = time.time()
        
        suffix="_2_stage" + "_" + str(evaluation) + "_warmup"
        suffix_dir = "_ablation_warmup_DGEMO"
        
        if not os.path.exists(f"logs_{test_ins}{suffix_dir}"):
            os.makedirs(f"logs_{test_ins}{suffix_dir}")
            
        n_sample = 5 
        n_iter = int((max_evaluation - evaluation)/n_sample)
            
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
        
        pref_vec_test = sampling_vector_evenly(n_obj, n_test)
        
        print(f"Problem: {test_ins}\nN dim: {n_dim} \nN objective: {n_obj} \nLogs dir: logs_{test_ins}{suffix_dir}/\nRun: {suffix}")
        print(f"Number of warm-up evaluation: {evaluation}\nNumber of evaluation per iteration: {n_sample}")
        
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
        x_init = np.load(f"warmup_evaluation/DGEMO_{test_ins}_X_{n_dim}_220.npy")
        x_init = x_init[:evaluation]
        
        y_init = problem.evaluate(torch.from_numpy(x_init).to(device))
        
        X = x_init
        Y = y_init.cpu().numpy()
        # print(X, Y)

        z = torch.zeros(n_obj).to(device)
        
        psmodel = ParetoSetModel(n_dim, n_obj, bound)
        psmodel.to(device)
        beta = 0.5
                
        
        # n_iter batch selections 
        for i_iter in range(n_iter):
            print(f"Iteration:  {i_iter + 1 :03d}/{n_iter}  ||  Time: {(time.time() - start)/60:.2f} min")
            
            # $$\theta_i = (1-beta) * \theta_{i-1} + beta * Noise 
            # beta starts by 0.5 and reduce by 10^-1 for every 10 iterations
            if i_iter > 0:
                with torch.no_grad():
                    for params in psmodel.parameters():
                        noise = torch.empty(params.shape).to(device)
                        if noise.dim() > 1:
                            nn.init.xavier_uniform_(noise).to(device)
                            # \theta_i = beta*theta_{i-1} + (1-beta)* Noise
                            params.mul_(beta).add_(noise, alpha = 1 - beta)
            if ((i_iter + 1) %10) == 0:
                beta *= 0.1
            
            # optimizer
            optimizer = torch.optim.Adam(psmodel.parameters(), lr=1e-3)
            
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
                pref = sampling_vector_randomly(n_obj, n_region, n_region_vec)          
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
            pref = sampling_vector_randomly(n_obj, n_region, n_candidate_per_region)
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
                    # X_candidate_tch[X_candidate_tch <= 0]  = 0
                    # X_candidate_tch[X_candidate_tch >= 1]  = 1  
                    for i in range (n_dim):
                        X_candidate_tch[:,i] = np.clip(X_candidate_tch[:, i], a_min=problem.lbound[i], a_max = problem.ubound[i])

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

            with open(f"logs_{test_ins}{suffix_dir}/hv_{test_ins}_{n_dim}{suffix}.pkl", 'wb') as output_file:
                pickle.dump([hv_list], output_file)
            
            np.save(f"logs_{test_ins}{suffix_dir}/evaluation_{test_ins}_X_{n_dim}{suffix}", X)
            np.save(f"logs_{test_ins}{suffix_dir}/evaluation_{test_ins}_Y_{n_dim}{suffix}", Y)
            
            np.save(f"logs_{test_ins}{suffix_dir}/front_{test_ins}_{n_dim}{suffix}", front_list)
            np.save(f"logs_{test_ins}{suffix_dir}/x_{test_ins}_{n_dim}{suffix}", x_list)
            np.save(f"logs_{test_ins}{suffix_dir}/y_{test_ins}_{n_dim}{suffix}", y_list)