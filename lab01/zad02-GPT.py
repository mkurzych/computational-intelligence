import random
import math
import matplotlib.pyplot as plt
import numpy as np


def projectile_motion(v0, theta, h0, g=9.81, dt=0.01):
    """
    Compute the trajectory of a projectile motion with a given starting height.

    Parameters:
        v0 (float): Initial velocity magnitude (m/s).
        theta (float): Launch angle in radians.
        h0 (float): Starting height (m).
        g (float): Acceleration due to gravity (m/s^2), default is 9.81 m/s^2.
        dt (float): Time step (s), default is 0.01 s.

    Returns:
        Tuple of arrays (x, y) containing the x and y coordinates of the projectile.
    """
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

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


def aim_trebuchet(target_distance, initial_height=100, initial_velocity=50, gravity=9.81):
    """
    Simulate aiming a trebuchet to hit a target.

    Parameters:
        target_distance (float): Distance to the target (m).
        initial_height (float): Initial height of the projectile (m), default is 100 m.
        initial_velocity (float): Initial velocity of the projectile (m/s), default is 50 m/s.
        gravity (float): Acceleration due to gravity (m/s^2), default is 9.81 m/s^2.

    Returns:
        Tuple of angle (radians) and number of tries.
    """
    missed = True
    tries = 0

    while missed:
        tries += 1
        angle_deg = float(input("Input angle in degrees: "))
        angle_rad = math.radians(angle_deg)
        distance = ((initial_velocity * math.sin(angle_rad)) + math.sqrt(
            (initial_velocity ** 2 * math.sin(angle_rad) ** 2) + (2 * gravity * initial_height))) * (
                               (initial_velocity * math.cos(angle_rad)) / gravity)

        print("Your distance is", distance)

        if target_distance - 5 < distance < target_distance + 5:
            missed = False
            print("Target is hit!")
            print("You took", tries, "tries")

    return angle_rad, tries


# Define initial conditions
target_distance = random.randint(50, 340)

print("Your target is", target_distance, "m away")

# Aim the trebuchet
angle, tries = aim_trebuchet(target_distance)

# Compute and plot the trajectory
x, y = projectile_motion(50, angle, 100)
plt.plot(x, y, label='Projectile Motion')
plt.title("Projectile Motion for the Trebuchet")
plt.xlabel("Distance (m)")
plt.ylabel("Height (m)")
plt.grid(True)
plt.legend()
plt.savefig('trajectory_GPT.png')  # Save the plot as a PNG image
plt.show()  # Display the plot
