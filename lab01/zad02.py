import random
import math
import matplotlib.pyplot as plt
import numpy as np


def projectile_motion(v0, theta, h0, g=9.81, dt=0.01):
    theta_rad = np.deg2rad(theta)
    vx0 = v0 * np.cos(theta_rad)
    vy0 = v0 * np.sin(theta_rad)

    x = [0]
    y = [h0]
    vx = [vx0]
    vy = [vy0]
    t = [0]

    while y[-1] >= 0:
        t.append(t[-1] + dt)
        x.append(x[-1] + vx[-1] * dt)
        y.append(y[-1] + vy[-1] * dt)
        vx.append(vx[-1])
        vy.append(vy[-1] - g * dt)

    return np.array(x), np.array(y)


d = random.randint(225, 340)
v = 50
h = 100
a = 0
distance = 0
print("Your target is", d, "m away")

g = 9.81
missed = True
i = 0

while missed:
    i += 1
    a = float(input("Input angle: "))
    a = math.radians(a)
    distance = ((v * math.sin(a)) + math.sqrt((math.pow(v, 2) * math.pow(math.sin(a), 2)) + (2 * g * h))) * ((v * math.cos(a)) / g)
    print("Your distance is", distance)
    if d - 5 < distance < d + 5:
        missed = False
        print("Target is down!")
        print("You took", i, "tries")

x, y = projectile_motion(v, a, 100)
plt.plot(x, y)
plt.title("Projectile Motion for the Trebuchet")
plt.ylabel("Height (m)")
plt.xlabel("Distance (m)")
plt.grid(True)
plt.savefig('trajectory.png')
plt.show()
