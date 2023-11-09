from numpy.polynomial import Polynomial
import numpy as np
import math


# Function to check Log base 2
def Log2(x):
    return (math.log10(x) / math.log10(2))
# Function to check if x is power of 2
def isPowerOfTwo(n):
    return (math.ceil(Log2(n)) == math.floor(Log2(n)))


def divide(vectorA, vectorB):
    '''A is a vector of length n, and B is a vector of length m where m ≤ n
    representing polynomials A(x) and B(x) where A(x) = B(x) * C(x) Output is
    the vector representation of C(x)'''
    #Pad 0’s to make both A and B to have the same length which is a power of 2.
    while not isPowerOfTwo(len(vectorA)):
        vectorA.append(0)
    while not isPowerOfTwo(len(vectorB)):
        vectorB.append(0)
    
    f1 = FFT(A)
    f2 = FFT(B)
    
    #Initialize a vector f3 of length t = |f1|.
    t = len(f1)
    f3 = [0] * t
    for j in range(t-1): 
        # THIS LINE IS NOT DONE YET Warning: if f2[j] = 0, then restart by randomly adding a very small constant to some coefficients of A[x]
        f3[j] = f1[j] / f2[j] 
    f4 = IFFT(f3)
    #Trim 0s from f4 and output.
    f4 = np.trim_zeros(f4)
    return f4