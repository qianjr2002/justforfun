import random
import numpy as np

n= 300

def marline(n): 
    #n为天数
    wea = [];
    # a = 1 if random.random()<=0.5 else 0;
    wea.append(1 if random.random()<=0.5 else 0)
    #第一天有0.5概率为晴，0.5概率为雨
    temp = -1
    for i in range(1,n):
        if (wea[i-1]==1):
            temp = 1 if random.random()<=0.7 else 0
            #晴天变晴天概率为 0.7
            #晴天变雨天概率为 0.3
        if(wea[i-1]==0):
            temp = 1 if random.random()<=0.4 else 0
            #雨天变晴天概率为 0.4
            #雨天变雨天概率为 0.6
        wea.append(temp)
    return wea

print(marline(n))

def act(n):
    act = []
    act.append(1 if random.random()<=0.5 else 0)
    #第一天的动作随机
    temp =-1
    for i in range(1,n):
        
        if (marline(n)[i-1]==1):
            temp = 1 if random.random()<=0.1 else 0
            #昨天晴天今天打伞概率为 0.1//p1
            #昨天晴天今天不打伞概率为 0.9//1-p1
        if(marline(n)[i-1]==0):
            temp = 1 if random.random()<=0.7 else 0
            #昨天雨天今天打伞概率为 0.7//p2
            #昨天雨天今天不打伞概率为 0.3//1-p2
        act.append(temp)
    return act
print(act(n))

def pointGet(n):
    # weather = arr(0.4,n)
    # act = arr(0.5,n)
    point = np.zeros(n)

    for i in range(n):
        if (marline(n)[i]==1):
            if( act(n)[i]==1):
                point[i] = -0.5
            else:
                point[i] = 1
        else:
            if (act(n)[i]==1):
                point[i] = 1
            else:
                point[i] = -1
    # print(point)
    aver = np.average(point)
    return aver
print(pointGet(n))