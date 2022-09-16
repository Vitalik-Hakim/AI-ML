import numpy as np

arr = np.array([np.nan, 1, 2])
print(repr(arr))

arr = np.array([np.nan, 'abc'])
print(repr(arr))

#Will result in a ValueError: If we uncomment line 8 and run again. / because np.nan cannot take on an integer type.
#np.array([np.nan, 1, 2], dtype=np.int32)
np.array([np.nan, 1, 2], dtype=np.float32)

