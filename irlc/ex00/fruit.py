"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
fruitPrices = {'apples': 2.00, 'oranges': 1.50, 'pears': 1.75} 

def buyFruit(fruitPrices, fruit, numPounds):
    if fruit not in fruitPrices:
        # This is the easiest way to format text. Notice the f in the front.
        print(f"Sorry we don't have {fruit}")
    else:
        cost = fruitPrices[fruit] * numPounds
        print(f"That'll be {cost} please")


if __name__ == '__main__':  # True when script is run
    buyFruit(fruitPrices, 'apples', 2.4) 
    buyFruit(fruitPrices, 'coconuts', 2)  
