import pandas as pd
import numpy as np

missing_values = ("n/a", "na", "--", '-')
irises = pd.read_csv("iris_with_errors.csv", na_values=missing_values)
print(irises.head())

# a) Policz ile jest w bazie brakujących lub nieuzupełnionych danych.
# Wyświetl statystyki bazy danych z błędami.

print(irises.isnull().sum().sum())
print(irises.isnull().sum())

# b) Sprawdź, czy wszystkie dane numeryczne są z zakresu (0; 15).
# Dane spoza zakresu muszą być poprawione. Możesz tutaj użyć metody:
# za błędne dane podstaw średnią (lub medianę) z danej kolumny.


def fix_outliers(df, column):
    median = df[column].median()
    df[column] = df[column].apply(lambda x: median if x < 0 or x > 15 else x)


print(irises[91:96])

fix_outliers(irises, "sepal.length")
fix_outliers(irises, "sepal.width")
fix_outliers(irises, "petal.length")
fix_outliers(irises, "petal.width")

print(irises[91:96])

# c) Sprawdź, czy wszystkie gatunki są napisami: „Setosa”, „Versicolor”
# lub „Virginica”. Jeśli nie, wskaż, jakie popełniono błędy i popraw
# je własną (sensowną) metodą.


def fix_species(item):
    if item == "setosa":
        return "Setosa"
    if item == "Versicolour":
        return "Versicolor"
    if item == "virginica":
        return "Virginica"
    return item


print(irises["variety"].unique())
irises["variety"] = irises["variety"].apply(fix_species)
print(irises["variety"].unique())
