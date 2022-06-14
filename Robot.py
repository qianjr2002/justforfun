import numpy as np
import random

n = 1000
def arr(p,n):
    #p是为1的概率，n是长度
    arr = []
    for i in range(n):
        arr.append(1 if random.random()<=p else 0)
    return arr
def pointGet(n):
    weather = arr(0.4,n)    
    # weather==1-->晴天,weather==0-->雨天
    act = arr(0.5,n)
    #act==1-->打伞,act==0-->不打伞
    point = np.zeros(n)

    for i in range(n):
        if (weather[i]==1):
            if( act[i]==1):
                point[i] = -0.5
            else:
                point[i] = 1
        else:
            if (act[i]==1):
                point[i] = 1
            else:
                point[i] = -1
    # print(point)
    aver = np.average(point)
    return aver
print(pointGet(n))