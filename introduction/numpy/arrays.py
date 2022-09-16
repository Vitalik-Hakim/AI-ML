import numpy as np  # import the NumPy library

arr = [-1, 2, 5]
# Initializing a NumPy array
arr = np.array(arr, dtype=np.float32) # with a specific data type
arr_mixed = [1,"7",2.3]
arr = np.array(arr_mixed) # this can be mixed data types

# Print the representation of the array
print(repr(arr))
print(repr(arr_mixed)) # UPCAST 