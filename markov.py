import numpy as np
import random as rm
import markov

# 状态空间
states = ["sunny","rainy"]
 
# 可能的事件序列
transitionName = [["SS","SR",],["RS","RR",]]
 
# 概率矩阵（转移矩阵）
transitionMatrix = [[0.7,0.3],[0.4,0.6]]

if sum(transitionMatrix[0])+sum(transitionMatrix[1]) != 2:
    print("Somewhere, something went wrong. Transition matrix, perhaps?")
else: print("All is gonna be okay, you should move on!! ")

# 实现了可以预测状态的马尔可夫模型的函数。
def activity_forecast(days):
    # 选择初始状态
    activityToday = "sunny"
    print("Start state: " + activityToday)
    # 应该记录选择的状态序列。这里现在只有初始状态。
    activityList = [activityToday]
    i = 0
    # 计算 activityList 的概率
    prob = 1
    while i != days:
        if activityToday == "sunny":
            change = np.random.choice(transitionName[0],replace=True,p=transitionMatrix[0])
            if change == "SS":
                prob = prob * 0.7
                activityList.append("sunny")
                # pass
            else:
                prob = prob * 0.3
                activityList.append("rainy")
        elif activityToday == "rainy":
            change = np.random.choice(transitionName[1],replace=True,p=transitionMatrix[1])
            if change == "RS":
                prob = prob * 0.4
                activityList.append("sunny")
                # pass
            else:
                prob = prob * 0.6
                activityList.append("rainy")
        i += 1  
    print("Possible states: " + str(activityList))
    print("End state after "+ str(days) + " days: " + activityToday)
    print("Probability of the possible sequence of states: " + str(prob))
 
# 预测 2 天后的可能状态
activity_forecast(2)
