# import math
import numpy as np
# exp = math.e

layer_outputs = [4.8, 1.21, 2.385]
exp_values = np.exp(layer_outputs)

# for output in layer_outputs:
#     exp_values.append(exp**output)

# total_exp = sum(exp_values)

norm_values = exp_values / np.sum(exp_values)

# for value in exp_values:
#     norm_values.append(value / total_exp)

print(norm_values)

print(sum(norm_values))