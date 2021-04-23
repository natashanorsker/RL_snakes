"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc.ex00.shop import FruitShop 

fruitPrices = {'apples': 1.00, 'oranges': 1.50, 'pears': 1.75}
shopA = FruitShop(name='the Berkeley Bowl', fruitPrices=fruitPrices)
applePrice = shopA.getCostPerPound('apples')
print(f'Apples cost {applePrice} at {shopA.name}')

otherFruitPrices = {'kiwis': 6.00, 'apples': 4.50, 'peaches': 8.75}
shopB = FruitShop(name='the Stanford Mall', fruitPrices=otherFruitPrices)
otherPrice = shopB.getCostPerPound('apples')
print(f'Apples cost {otherPrice} at {shopB.name}')
print("My, that's expensive!") 
