"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
fruits = ['apples', 'oranges', 'pears'] 

# Iterate i=0, 1, 2, ...
for i in range(len(fruits)):
    print(i, fruits[i])
print("---")
# Iterate over index and value using enumerate:
for i, fruit in enumerate(fruits):
    print(i, fruit) 
