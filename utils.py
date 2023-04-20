'''
FILE (minus some comments) FROM CEDEGAO ET AL 2021
'''
# basic functions used across models
import numpy as np # for numeric calculations
import pandas as pd # for python data 

def getSizeOfNestedList(listOfElem):
    ''' Get number of elements in a nested list'''
    count = 0
    # Iterate over the list
    for elem in listOfElem:
        # Check if type of element is list
        if type(elem) == list:
            # Again call this function to get the size of this element
            count += getSizeOfNestedList(elem)
        else:
            count += 1
    return count
def unique(l):
    return np.array(list(set(l)))
    
def get_AIC(bestllh, NParam):
    return 2 * bestllh + 2 * NParam
def get_BIC(bestllh, NParam, nSample):
    return 2 * bestllh + np.log(nSample) * NParam

def np2pd(Data, colnames): #Top: numpy to panda
    return pd.DataFrame(Data, columns=colnames)
def pd2np(Data): #Top: panda to numpy
    return Data.to_numpy()

