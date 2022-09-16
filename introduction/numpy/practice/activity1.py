import numpy as np


# The first array we'll create comes straight from a list of integers and np.nan. The list contains np.nan as the first element, and the integers from 2 to 5, inclusive, as the next four elements.

# Set arr equal to np.array applied to the specified list.

arr = np.array([np.nan,2,3,4,5])

# We now want to copy the array so we can change the first element to 10. This way we don't modify the original array.

# Set arr2 equal to arr.copy(), then set the first element of arr2 equal to 10.

arr2 = np.copy(arr)

arr2[0] = 10

print(repr(arr2))

float_arr = np.array([1, 5.4, 3])
print(repr(float_arr))

float_arr2 = arr2.astype(np.float32)

print(float_arr2)

matrix = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)