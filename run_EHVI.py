import numpy as np
import torch
import pickle
import os
import pdb
from itertools import combinations
import argparse
from problem import get_problem

from lhs import lhs
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from mobo.surrogate_model import GaussianProcess
from mobo.transformation import StandardTransform


from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    )
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
    )
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.sampling import sample_simplex
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch import fit_gpytorch_mll

from pymoo. config import Config
Config.warnings['not_compiled'] = False

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

parser = argparse.ArgumentParser()
parser.add_argument("--n_dim", type=int, default=6)
parser.add_argument("--n_iter", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--problem", type=str, default="DTLZ2")
args = parser.parse_args()

n_dim = args.n_dim
n_iter = args.n_iter
test_ins = args.problem
BATCH_SIZE = args.batch_size

NUM_RESTARTS = 5 # if not SMOKE_TEST else 2
RAW_SAMPLES = 5 # if not SMOKE_TEST else 5
# N_BATCH = 10 # if not SMOKE_TEST else 10
MC_SAMPLES = 16 # if not SMOKE_TEST else 16


def initialize_model(train_x, train_obj, bounds, n_obj):
    # train_x = normalize(train_x, bounds)
    models = []
    NOISE_SE = 1.e-7 * torch.ones(n_obj, **tkwargs)
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i : i + 1]
        train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)
        models.append(SingleTaskGP( 
              # FixedNoiseGP(
                train_x, train_y,# train_yvar, outcome_transform=Standardize(m=1)
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def optimize_qehvi_and_get_observation(model, ref_point, train_obj, sampler,standard_bounds, BATCH_SIZE=BATCH_SIZE, NUM_RESTARTS=NUM_RESTARTS,RAW_SAMPLES=RAW_SAMPLES):
    
    with torch.no_grad():

        partitioning = FastNondominatedPartitioning(
        ref_point=ref_point,
        Y=train_obj,
    )
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 100000, "maxiter": 10000},
        sequential=True,
    )
    # observe new values
    # new_x = unnormalize(candidates.detach(), bounds=standard_bounds)
    new_x = candidates
    new_obj_true = - problem.evaluate(new_x) + 1
    return new_x, new_obj_true, acq_func(new_x)
# n_run = 2 #20 
# number of initialized solutions
n_init = 20 
# number of iterations, and batch size per iteration


# device
device = 'cuda:2' # 'cuda:1'


# the dimension of optimized variables
print(f"Problem: {test_ins}\n N dim: {n_dim}")
print(f"Init evaluation: {n_init}\n N iteration: {n_iter} \n Batch size: {BATCH_SIZE}")

problem = get_problem(test_ins, n_dim=n_dim)
bound = problem.bound
if bound > 0:
    problem.ubound *= bound
    problem.lbound *= bound
hv_all_value = []
n_dim = problem.n_dim
n_obj = problem.n_obj

ref_point = problem.nadir_point
ref_point = [1.1*x for x in ref_point]

# x_init = lhs(n_dim, n_init)
if bound == 0:
    x_init =  lhs(n_dim, n_init)
else:
    x_init =  lhs(n_dim, n_init) * 2 * bound - bound

y_init = - problem.evaluate(torch.from_numpy(x_init).to(device)) + 1

# print(x_init)
X = x_init
Y = y_init.cpu().numpy()

z = torch.zeros(n_obj).to(device)
for i_iter in range(n_iter):
    qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
    train_x = torch.tensor(X)
    train_obj = torch.tensor(Y)
    bounds = torch.cat([problem.lbound,problem.ubound]).reshape(-1,n_dim)
    # print(bounds)
    # exit()
    mll_qehvi, model_qehvi = initialize_model(train_x, train_obj, bounds, n_obj)
    fit_gpytorch_mll(mll_qehvi)
    
    # pdb.set_trace()

    partitioning = FastNondominatedPartitioning(
        ref_point=torch.tensor(ref_point, dtype = torch.float64),
        Y=train_obj,
    )
    # acq_func = qExpectedHypervolumeImprovement(
    #     model=model_qehvi,
    #     ref_point=torch.tensor(ref_point),
    #     partitioning=partitioning,
    #     sampler=qehvi_sampler,)


    X_new, Y_new, acc_value = optimize_qehvi_and_get_observation(
       model_qehvi, torch.tensor(ref_point, dtype = torch.float64), train_obj, qehvi_sampler,bounds)
    print(X_new.shape, X_new.clone().detach().cpu().numpy())
    X = np.vstack([X,X_new.detach().cpu().numpy()])
    Y = np.vstack([Y,Y_new.detach().cpu().numpy()])
    
    # check the current HV for evaluated solutions
    hv = HV(ref_point=1.1*np.ones(n_obj))
    hv_value = hv(-Y + 1)
    hv_all_value.append(hv_value)
    print(f"iter {i_iter}","hv", "{:.2e}".format(np.mean(hv_value)), "acc_value", acc_value)

print("************************************************************")
suffix = n_init + BATCH_SIZE * n_iter
suffix_dir = ""
save_dir = f"warmup_evaluation"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(f"{save_dir}/EHVI_{test_ins}_X_{n_dim}_{suffix}_{BATCH_SIZE}.npy", X)
np.save(f"{save_dir}/EHVI_{test_ins}_Y_{n_dim}_{suffix}_{BATCH_SIZE}.npy", - Y + 1)
# with open(f"{save_dir}/EHVI_warmup_{test_ins}_{n_dim}_{suffix}.pickle", 'wb') as output_file:
#     pickle.dump([hv_all_value], output_file)
print(f"Save file at `{save_dir}/` ")