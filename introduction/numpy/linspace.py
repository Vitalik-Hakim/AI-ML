import numpy as np


arr = np.linspace(1, 10, num=5) # add last number
print(repr(arr))

arr = np.linspace(5, 11, num=4, endpoint=False) # don't add the last number
print(repr(arr))

arr = np.linspace(5, 11, num=4, dtype=np.int32) # specify datatype
print(repr(arr))