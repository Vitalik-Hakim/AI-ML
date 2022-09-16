import numpy as np 



# obj 1
arr = np.arange(12)
reshaped = np.reshape(arr,(2, 3, 2))
#obj 2
flattened = reshaped.flatten()
transposed = np.transpose(reshaped, axes=(1, 2, 0))
# obj 3
zeros_arr = np.zeros(5)
ones_arr = np.ones_like(transposed)
#obj 4
points = np.linspace(-3.5,1.5, num=101,)