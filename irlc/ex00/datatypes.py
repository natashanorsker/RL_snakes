"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
def lists():
    a = [1, 2, 3, 10, 16]
    print("List a is", a)
    a_reverse = a.reverse()
    print("reverse of a", a_reverse)
    a_squared = [x**2 for x in a]
    print("squared elements of a", a_squared)

def dicts():
    a = {'a': 1, 'b': 2, 'c': 3}
    a['d'] = 4 # add a key 'd' to a with value 4
    for k in a:
        print(k) # print all the keys

    for k, v in a.items():
        print(k, 'has value', v)

    # make a new dictionary with the same keys as a but the values are squared
    b = {k: v**2 for k, v in a.items()}
    print(b)

if __name__ == "__main__":
    lists()
    dicts()
