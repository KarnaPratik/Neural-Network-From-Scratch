import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights1 =  [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
# Both inputs and weights are same shpae 3 X 4, so matrix multiplication is not possible, np.dot attempts, 1 times 0.2 plus 2 times 0.5 plus 3 times -0.26 and then 2.5 times nothing but that's not what we want. So, also chaning np.dot to(weights, inputs) will still throw the same error. To solve this we can simply transpose weights matrix. So now it becomes a 4 X 3 matrix and multiplication is possible. Weights is just a list of lists, so first we convert it to arrays so transpose is possible.

# Note: Let's say we have only a 1D array or a simple list like bias, now it is not rows by columns, it's just (3,). Eg: Input list having 4 values is of order (4,). So initially np.dot(weights, inputs) would work becuase 3 X 4 and (4,), indices 4 match. But in batches, matrix order are to be taken into account.

bias = [2, 3, 0.5]
output = [0, 0, 0]

layer1_output = np.dot(inputs, np.array(weights1).T) + bias


print(layer1_output)

weights2 =  [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

bias2 = [-1, 2, -0.5]

layer2_output = np.dot(layer1_output, np.array(weights2).T) + bias2

print(layer2_output)