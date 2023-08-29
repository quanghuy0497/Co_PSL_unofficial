import numpy as np
from pymoo.util.ref_dirs import get_reference_directions


ref_dirs = get_reference_directions("uniform", 2, n_partitions=400)
print(ref_dirs.shape)

ref_dirs = get_reference_directions("uniform", 3, n_partitions=30)
print(ref_dirs.shape)