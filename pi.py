import turtle as t
import random
import matplotlib.pyplot as plt
import math

t.speed(0)
t.up()
t.setposition(-100,-100)
t.down()
for i in range(4):
    t.fd(200)
    t.left(90)
t.up()
t.setposition(0,-100)
t.down()
t.circle(100)
# a=input()
t.ht()
in_circle = 0
out_circle = 0
pi_values = []
avg_pi_errors = []

for i in range(10):
    for j in range(1000):
        x = random.randrange(-100,100)
        y = random.randrange(-100,100)
        if(x**2+y**2>100**2):
            t.color("black")
            t.up()
            t.goto(x,y)
            t.down()
            t.dot()
            out_circle += 1
        else:
            t.color("red")
            t.up()
            t.goto(x,y)
            t.down()
            t.dot()
            in_circle += 1
        pi = 4.0*in_circle/(in_circle+out_circle)
        pi_values.append(pi)
        avg_pi_errors = [abs(math.pi - pi) for pi in pi_values]
    print(pi_values[-1])

plt.axhline(y=math.pi,color="g",linestyle="-")
plt.plot(pi_values)
plt.xlabel("Iterations")
plt.ylabel("Value of PI")
plt.show()

plt.axhline(y=0.0,color="g",linestyle="-")
plt.plot(avg_pi_errors)
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.show()