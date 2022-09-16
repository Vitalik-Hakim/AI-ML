import numpy as np


# To represent infinity in NumPy, we use the np.inf special value. We can also represent negative infinity with -np.inf.

# The code below shows an example usage of np.inf. Note that np.inf cannot take on an integer type



print(np.inf > 1000000) # returns true

arr = np.array([np.inf, 5])
print(repr(arr))

arr = np.array([-np.inf, 1])
print(repr(arr))

# Will result in a OverflowError: If we uncomment line 10 and run again.
#np.array([np.inf, 3], dtype=np.int32)
np.array([np.inf, 3], dtype=np.float32)