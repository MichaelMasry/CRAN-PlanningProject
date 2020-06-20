import numpy as np
from _collections import deque


def shiftbit(array,direction):
    items=deque(array)
    items.rotate(direction)
    return items


instalM=np.zeros((10,6))

instalM[:,0]=1
neighbor_combinations = np.empty((600), dtype=object)

x=np.reshape(instalM[:,5],np.size(instalM,0))
print(x)
def recersive(array,i,j):                               ##############hint starts when called from maxrow-1 aka last row index
    check=np.reshape(array[:,5],np.size(array,0))
    if(np.all(check)==1):
        return neighbor_combinations                       ###stop when all the valuse of the matrix on the rightmost are 1's
    if(j==5):
        recersive(array,i-1,0)
    array[i]=shiftbit(array[i],1)
    np.concatenate(neighbor_combinations,array.copy())        ##### better 2 be done with count i made with conc 2 revie my function point
    recersive(array,i,j+1)                                    #####try on small matrix 2 get the idea