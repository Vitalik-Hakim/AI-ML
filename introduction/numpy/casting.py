import numpy as np


arr = np.array([0, 1, 2])
print(arr.dtype)
arr_clone = arr.astype(np.float32)
print(arr_clone.dtype)

# output
# int32
# float32