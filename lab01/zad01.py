import datetime
import math

name = input("Enter your name: ")
year = int(input("Enter your birth year: "))
month = int(input("Enter your birth month: "))
day = int(input("Enter your birth day: "))

date = datetime.date(year, month, day)
delta = datetime.date.today() - date

print("\nHi " + name + "!")
print("Today is your " + str(delta.days) + "th day of being alive")

physical = math.sin((math.pi * 2 * delta.days) / 23)
emotional = math.sin((math.pi * 2 * delta.days) / 28)
intellectual = math.sin((math.pi * 2 * delta.days) / 33)

print("\nYour physical wave: " + str(physical))
print("Your emotional wave: " + str(emotional))
print("Your intellectual wave: " + str(intellectual))

if physical > 0.5 or emotional > 0.5 or intellectual > 0.5:
    print("Your having a great day! Congrats")
if physical < -0.5 or emotional < -0.5 or intellectual < -0.5:
    print("Sorry that you're not feeling well :(")
    t_physical = math.sin((math.pi * 2 * (delta.days + 1)) / 23)
    t_emotional = math.sin((math.pi * 2 * (delta.days + 1)) / 28)
    t_intellectual = math.sin((math.pi * 2 * (delta.days + 1)) / 33)
    if physical < t_physical or emotional < t_emotional or intellectual < t_intellectual:
        print("But don't worry, tomorrow will be better")

# 30 min

