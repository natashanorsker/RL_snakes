"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc.ex00.shop import FruitShop  
from irlc.ex00.shop_test import fruitPrices

class FancyFruitShop(FruitShop):
    def __init__(self, name, fruitPrices):
        super().__init__(name, fruitPrices)  # call super-class __init___
        print("(This is a fine establishment)")

    def getCostPerPound(self, fruit):  # Using inline if for brevity
        return 2 * self.fruitPrices[fruit] if fruit in self.fruitPrices else None

if __name__ == "__main__":
    shopA = FruitShop("The Brown Banana", fruitPrices) 
    shopB = FancyFruitShop("The Golden Delicious", fruitPrices)

    print("Shopping at: ", shopA.getName(), "price of apples", shopA.getCostPerPound('apples'))
    print("Shopping at: ", shopB.getName(), "price of apples", shopB.getCostPerPound('apples')) 
