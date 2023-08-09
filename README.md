#### Code (run with dim 5):
+ Run Original PSL: `python run_original.py --n_dim 5`
+ Run PSL with Initialize theta only: `python run_init.py --n_dim 5`
+ Warming up EHIV (default `20 iteration`): `python run_EHVI_warmup.py --n_dim 5 --n_iter 20`
+ Run PSL with warming up (required warming up EHIV beforehand): `python run_warmup.py --n_dim 5`
+ Run PSL with 2-stage framework (required warming up EHIV beforehand): `python run_warmup_init.py --n_dim 5`

**Notes**:
  + Default MOO **problem** is `Cybenko`, change the problem in each code into the one appropriate with your test beforhand
  + The **constrained** for PSL is already set to be in `[-5, 5]`. If you want to change the boundary, please change the variable `bound = 5` within the code
  + The code is already embedded with **partitioning** in preference vector samples, this does not affect the performace. The details of partitioning are as follows:
    + *Training*:dividing the preference space into 10 equally regions, and sample randomly 2 pref_vec each region for each PSL training iteration
    + *Testing*: 50 evenly distributed pref vectors for testing
  + **Log dir** is automatically generated with name `logs_{problem_name}`
  + All the PSLs are run with `50` **iterations**, please don't change that for a fair observation
  + Result plot is in `plot_warmup_init.ipynb`
    + Each PSL codes is attained with a suffix index for logging (i.e., `_original`, `_warmup_init`,...). Just notice that for ploting results
    + You can modify the plotting code on your own (for iteration visualization, etc). However, please follow the same plotting format for the convenient in comparisions.
