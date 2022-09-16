import numpy as np
a = np.array([0, 1])
b = np.array([9, 8])
c = a
print('Array a: {}'.format(repr(a)))
c[0] = 5
print('Array a: {}'.format(repr(a)))

d = b.copy()
d[0] = 6
d[1] = 9
#d[2] = 0 # this won't work because the matrix is less
print('Array b: {}'.format(repr(d)))

# essentially you can inherit a list by just reassigning it to another number