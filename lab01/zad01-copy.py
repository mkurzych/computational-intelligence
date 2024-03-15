import datetime
import math


def get_biorhythm(day):
    physical = math.sin((2 * math.pi * day) / 23)
    emotional = math.sin((2 * math.pi * day) / 28)
    intellectual = math.sin((2 * math.pi * day) / 33)
    return physical, emotional, intellectual


def print_biorhythm(name, delta):
    physical, emotional, intellectual = get_biorhythm(delta.days)
    print("\nHi {}!".format(name))
    print("Today is your {}th day of being alive".format(delta.days))
    print("\nYour physical wave: {:.2f}".format(physical))
    print("Your emotional wave: {:.2f}".format(emotional))
    print("Your intellectual wave: {:.2f}".format(intellectual))
    return physical, emotional, intellectual


def print_day_status(physical, emotional, intellectual, delta):
    if any(val > 0.5 for val in [physical, emotional, intellectual]):
        print("You're having a great day! Congrats")
    elif any(val < -0.5 for val in [physical, emotional, intellectual]):
        print("Sorry that you're not feeling well :(")
        t_physical, t_emotional, t_intellectual = get_biorhythm(delta.days + 1)
        if physical < t_physical or emotional < t_emotional or intellectual < t_intellectual:
            print("But don't worry, tomorrow will be better")


def main():
    name = input("Enter your name: ")
    year = int(input("Enter your birth year: "))
    month = int(input("Enter your birth month: "))
    day = int(input("Enter your birth day: "))

    birth_date = datetime.date(year, month, day)
    delta = datetime.date.today() - birth_date

    physical, emotional, intellectual = print_biorhythm(name, delta)
    print_day_status(physical, emotional, intellectual, delta)


if __name__ == "__main__":
    main()
