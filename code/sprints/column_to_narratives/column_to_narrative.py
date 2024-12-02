# [2412]
#

import pandas as pd

if False:
    print("CCC - Column to Narrative")

name_csv = "wikipedia_list_ccc.csv"

df = pd.read_csv(name_csv)


# For a given numeric column, compute a subset of descriptive statistics
#
smallest = df['enrollment'].min()
largest = df['enrollment'].max()
ave = int(df['enrollment'].mean())
median = int(df['enrollment'].median())
q_lo = df['enrollment'].quantile(0.1)
q_hi = df['enrollment'].quantile(0.9)
sum_ = df['enrollment'].sum()


# Generate narrative text per row based on the above statistics, assume each
# row has a unqiue name
# 
for index, row in df.iterrows():
    print('The ' + row['name'] + 
          ' had a fall 2023 student enrollment of ' +
          str(row['enrollment']) + '.')

    if row['enrollment'] < ave:
        print('The ' + row['name'] +
              'has a student enrollment that is smaller than the average enrollment (' +
              str(ave) + ').')
    else:
        print('The ' + row['name'] +
              'has a student enrollment that is larger than the average enrollment (' +
              str(ave) + ').')

    if row['enrollment'] <= q_lo:
        print('The ' + row['name'] +
              'has a student enrollment that is among the smallest in the state.')
    if row['enrollment'] >= q_hi:
        print('The ' + row['name'] +
              'has a student enrollment that is among the largest in the state.')

    if row['enrollment'] == smallest:
        print('The ' + row['name'] +
              'has a student enrollment that is the smallest in the state.')
    if row['enrollment'] == largest:
        print('The ' + row['name'] +
              'has a student enrollment that is the largest in the state.')

    print()


# generate overall narrative text for all schools, based on above statisttics
#
print('The average enrollment at a California community college is ' + str(ave) + '.')
print('The median enrollment at a California community college is ' + str(median) + '.')
print('The smallest enrollment at a California community college is ' + str(smallest) + '.')
print('The largest enrollment at a California community college is ' + str(largest) + '.')
print('The combined enrollment at all California community colleges is ' + str(sum_) + '.')

print('A table of California community colleges and their enrollments can be found ' +
      '[here](https://en.wikipedia.org/wiki/List_of_California_Community_Colleges).')

