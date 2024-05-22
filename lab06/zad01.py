import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

# Load data
df = pd.read_csv('titanic.csv')

# Drop unnecessary columns
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# Select all unique values from the dataset
items = set()
for col in df:
    items.update(df[col].unique())

# Encode the dataset
itemset = set(items)
encoded_vals = []
for index, row in df.iterrows():
    rowset = set(row)
    labels = {}
    uncommons = list(itemset - rowset)
    commons = list(itemset.intersection(rowset))
    for uc in uncommons:
        labels[uc] = False
    for com in commons:
        labels[com] = True
    encoded_vals.append(labels)

df = pd.DataFrame(encoded_vals)

# Get frequent items
freq_items = apriori(df, min_support=0.5, use_colnames=True, verbose=1)

# Get association rules
rules = association_rules(freq_items, metric="confidence", min_threshold=0.8)

# Display plots
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()

fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], fit_fn(rules['lift']))
plt.xlabel('lift')
plt.ylabel('confidence')
plt.title('Lift vs Confidence')
plt.show()

# Display rules
print(rules[(rules['confidence'] > 0.8) & (rules['support'] > 0.05)].sort_values(by='confidence', ascending=False))


#    antecedents    consequents  antecedent support  consequent support  \
# 4   (Male, No)        (Adult)            0.619718            0.950477
# 1         (No)        (Adult)            0.676965            0.950477
# 0       (Male)        (Adult)            0.786461            0.950477
# 3  (Adult, No)         (Male)            0.653339            0.786461
# 2         (No)         (Male)            0.676965            0.786461
# 5         (No)  (Adult, Male)            0.676965            0.757383

#     support  confidence      lift  leverage  conviction  zhangs_metric
# 4  0.603816    0.974340  1.025106  0.014788    1.929980       0.064404
# 1  0.653339    0.965101  1.015386  0.009900    1.419023       0.046906
# 0  0.757383    0.963027  1.013204  0.009870    1.339441       0.061028
# 3  0.603816    0.924200  1.175139  0.089991    2.817152       0.429921
# 2  0.619718    0.915436  1.163995  0.087312    2.525187       0.436144
# 5  0.603816    0.891946  1.177669  0.091095    2.245337       0.467023
