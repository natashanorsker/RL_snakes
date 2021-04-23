"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
nums = [1, 2, 3, 4, 5, 6] 
plusOneNums = [x + 1 for x in nums]

oddNums = [x for x in nums if x % 2 == 1]
print(oddNums)
oddNumsPlusOne = [x + 1 for x in nums if x % 2 == 1]
print(oddNumsPlusOne) 

""" 
Dictionary comprehension. We make a new dictionary where both the keys and values may be changed
"""
toy_cost = {'toy car': 10, 'skipping rope': 6, 'toy train': 20} 
print(toy_cost)

double_cost = {k: 2*v for k, v in toy_cost.items()}
print(double_cost)

bad_toys = {"broken "+k: v//2 for k, v in toy_cost.items()}
print(bad_toys)

expensive_toys = {k: v for k, v in toy_cost.items() if v >= 10}
print(expensive_toys) 
