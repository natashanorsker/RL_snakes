"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from unitgrade.unitgrade import QuestionGroup, Report, QPrintItem

class AdditionQuestion(QuestionGroup):  
    title = "Adding two numbers"
    class AdditionItem1(QPrintItem):
        a, b = 2, 2
        def compute_answer_print(self):
            from irlc.ex00.fruit_homework import add
            print(add(self.a, self.b)) 

    class AdditionItem2(AdditionItem1):
        a,b = -4, 2.45

class DataStructureQuestion(QuestionGroup):
    title = "Data structures"
    class PoliteAnimalsItem(QPrintItem):
        names = ["cat", "dog", "tiger"]
        def compute_answer_print(self):
            from irlc.ex00.fruit_homework import misterfy
            return misterfy(self.names)

    class PoliteAnimalsItem2(PoliteAnimalsItem):
        names = ["sea lion", "space sheep"]

    class PoliteAnimalsItem3(PoliteAnimalsItem):
        names = []

    class DieMeanitem(QPrintItem):
        p_die = {1: 0.20,
                 2: 0.10,
                 3: 0.15,
                 4: 0.05,
                 5: 0.10,
                 6: 0.40}

        def compute_answer_print(self):
            from irlc.ex00.fruit_homework import mean_value
            print(mean_value(self.p_die))

    class DieMeanitem2(DieMeanitem):
        p_die = {-10: 0.4,
                 200: 0.5,
                 30: 0.1}

    class FruitsOrderedItem(QPrintItem):
        order = {'apples': 1.0,
                 'oranges': 3.0}
        def compute_answer_print(self):
            from irlc.ex00.fruit_homework import fruits_ordered
            print(fruits_ordered(self.order))

    class FruitsOrderedItem2(FruitsOrderedItem):
        order = {102: "a high number",
                 23: "a less high number"}


class FruitShops(QuestionGroup):
    title = "Object-oriented programming: Finding best shop for fruit shopping"

    class FruitShopItem1(QPrintItem):
        def get_shops(self):
            from irlc.ex00.fruit_homework import OnlineFruitShop
            return [OnlineFruitShop('shop1',  {'apples': 2.0, 'oranges': 1.0}),
                    OnlineFruitShop('shop2', {'apples': 1.0, 'oranges': 5.0})]
        order = {'apples': 1.0, 'oranges': 3.0}

        def compute_answer_print(self):
            from irlc.ex00.fruit_homework import shopSmart
            print("Best shop", shopSmart(self.order, self.get_shops()).getName())

    class FruitShopItem2(FruitShopItem1):
        order = {'apples': 3.0}

    class FruitShopItem3(FruitShopItem1):
        def get_shops(self):
            from irlc.ex00.fruit_homework import OnlineFruitShop
            return [OnlineFruitShop('shop A', {'apples': 2.0, 'oranges': 1.0}),
                     OnlineFruitShop('shop B', {'apples': 1.0, 'oranges': 5.0}),
                     OnlineFruitShop('shop C',  {'apples': 1.5, 'oranges': 2.0})]
        order = {'apples':10.0, 'oranges': 3.0}


class FruitReport(Report):
    title = "Fruit and addition assignment"
    questions = [(AdditionQuestion, 1), (DataStructureQuestion, 1), (FruitShops, 2) ]
    import irlc
    pack_imports = [irlc]
    individual_imports = []

if __name__ == "__main__":
    from unitgrade.unitgrade_helpers import evaluate_report_student
    evaluate_report_student(FruitReport())
