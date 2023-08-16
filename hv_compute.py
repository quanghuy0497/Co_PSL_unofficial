import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import pickle
from pymoo.indicators.hv import HV
from pymoo.config import Config
Config.warnings['not_compiled'] = False
from problem import get_problem


methods = ['DGEMO', 'MOEAD-EGO', 'TS-TCH', 'TSEMO', 'USEMO-EI', 'PSL-MOBO', 'Co-PSL']

problem_names = ['DTLZ2', 'VLMOP2', 'F2', 'RE33', 'RE36', 'RE37']


for problem_name in problem_names:
    log_dir = f"weight/{problem_name}"

    n_dim = 6 if problem_name in ['F2', 'DTLZ2', 'VLMOP2'] else 4
    
    problem = get_problem(problem_name)
    ref_point = problem.nadir_point 
    ref_point = [1.1 * x  for x in ref_point]
    hv = HV(ref_point=np.array(ref_point))
    
    for method in methods:
        sol = np.load(f'{log_dir}/evaluation_{problem_name}_Y_{n_dim}_{method}.npy')
        print(f"{problem_name} - {method} - {sol.shape}")
        for i in range(sol.shape[0]): 
            hv_tmp = []
            for j in range(sol.shape[1]):
                hv_tmp.append(hv(sol[i][:j])) 
            hv_tmp = np.expand_dims(hv_tmp, 0)
            hv_result = hv_tmp if i == 0 else np.vstack([hv_result, hv_tmp])
        print(hv_result.shape)
        np.save(f'{log_dir}/HV_{problem_name}_Y_{n_dim}_{method}.npy', hv_result)    