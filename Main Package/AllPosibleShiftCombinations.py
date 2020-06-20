import numpy as np
from _collections import deque


def shiftbit(array,direction):
    items=deque(array)
    items.rotate(direction)
    return items


instalM=np.zeros((100,6))

instalM[:,0]=1
neighbor_combinations = np.empty((10**6), dtype=object)
stebn=instalM[0]
count=0

instalk=np.zeros((10,6))

def restben(array,maxrox):
    hight = np.size(array, 0)
    array[maxrox:hight,:]=stebn.copy()



def recersive(array,i,j):                               ##############hint starts when called from maxrow-1 aka last row index
    global count
    numi=i
    numj=j
    width=np.size(array,1)
    hight=np.size(array,0)
    check=np.reshape(array[:,width-1],hight)             ########checkc dimentions (1x height)
    if(np.all(check)==1):
        print("check")
        return neighbor_combinations                       ###stop when all the valuse of the matrix on the rightmost are 1's
    if((array[i][width-1])==1):
        print("Last one is 1")
        numi-=1
        recersive(array,numi,6)
    else:
        if(j==6):
            print("j==6")
            print(array[i])
            array[i]=shiftbit(array[i],1)
            neighbor_combinations[count]=array.copy()
            count+=1
            numi+=1
            restben(array,numi)
            recersive(array,hight-1,0)
        else:
            print("in Last else")
            array[i]=shiftbit(array[i],1)
            neighbor_combinations[count]=array.copy()                                     ##### better 2 be done with count i made with conc 2 revie my function point
            count=count+1
            numj+=1
            recersive(array,numi,numj)                                    #####try on small matrix 2 get the idea


recersive(instalM,np.size(instalM,0)-1,1)
print(instalM)