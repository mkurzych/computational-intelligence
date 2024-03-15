import math
import datetime


def calculate_biorhythm(birthdate, target_date):
    # Biorhythm periods in days
    physical_period = 23
    emotional_period = 28
    intellectual_period = 33

    # Calculate the number of days between birthdate and target_date
    delta_days = (target_date - birthdate).days

    # Calculate biorhythm values
    physical_value = math.sin(2 * math.pi * delta_days / physical_period)
    emotional_value = math.sin(2 * math.pi * delta_days / emotional_period)
    intellectual_value = math.sin(2 * math.pi * delta_days / intellectual_period)

    return physical_value, emotional_value, intellectual_value


def main():
    # Input birthdate
    birthdate_input = input("Enter birthdate (YYYY-MM-DD): ")
    birthdate = datetime.datetime.strptime(birthdate_input, "%Y-%m-%d").date()

    # Set target date to current date
    target_date = datetime.date.today()

    # Calculate biorhythm
    physical, emotional, intellectual = calculate_biorhythm(birthdate, target_date)

    # Output results
    print("Physical biorhythm:", physical)
    print("Emotional biorhythm:", emotional)
    print("Intellectual biorhythm:", intellectual)


if __name__ == "__main__":
    main()

# 10 min, chat się zawiesił lol
