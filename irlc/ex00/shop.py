"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
class FruitShop: 
    def __init__(self, name, fruitPrices):
        """
            name: Name of the fruit shop

            fruitPrices: Dictionary with keys as fruit
            strings and prices for values e.g.
            {'apples': 2.00, 'oranges': 1.50, 'pears': 1.75}
        """
        self.fruitPrices = fruitPrices
        self.name = name
        print(f'Welcome to {name} fruit shop')

    def getCostPerPound(self, fruit):
        """
            fruit: Fruit string
        Returns cost of 'fruit', assuming 'fruit'
        is in our inventory or None otherwise
        """
        if fruit not in self.fruitPrices:
            return None
        return self.fruitPrices[fruit]

    def getName(self):
        return self.name 


if __name__ == "__main__": 
    shopName = 'the Berkeley Bowl'
    fruitPrices = {'apples': 1.00, 'oranges': 1.50, 'pears': 1.75}
    berkeleyShop = FruitShop(shopName, fruitPrices)
    applePrice = berkeleyShop.getCostPerPound('apples')
    print(applePrice)
    print('Apples cost $%.2f at %s.' % (applePrice, shopName))

    otherName = 'the Stanford Mall'
    otherFruitPrices = {'kiwis': 6.00, 'apples': 4.50, 'peaches': 8.75}
    otherFruitShop = FruitShop(otherName, otherFruitPrices)
    otherPrice = otherFruitShop.getCostPerPound('apples')
    print(otherPrice)
    print('Apples cost $%.2f at %s.' % (otherPrice, otherName))
    print("My, that's expensive!") 
