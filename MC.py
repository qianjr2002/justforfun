import random
import numpy as np
import matplotlib.pyplot as plt

def coin_flip():
    return random.randint(0,1)
coin_flip()
list1=[]
def monte_carlo(n):
    results = 0
    for i in range(n):
        flip_result = coin_flip()
        results += flip_result
        prob_value = results/(i+1)
        list1.append(prob_value)
        plt.axhline(y=0.5,c="r",ls="-",lw=2)
        plt.xlabel("Iterations")
        plt.ylabel("Probability")
        plt.plot(list1)
    return results/n
answer = monte_carlo(500)
plt.show()
print("Fianl value:",answer)