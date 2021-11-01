sortdict = {
    0: "harten",
    1: "ruiten",
    2: "schoppen",
    3: "klaveren",
    4: "joker",
    5: "boer"
}


class Card:
    def __init__(self, number=None, sort=None):
        if number is not None:
            self.number = number
            if self.number < 48:
                self.sort = sortdict[int((self.number) / 12)]
                self.truenumber = self.number % 12 + 1
            elif self.number == 48:
                self.sort = sortdict[5]
                self.truenumber = 13
            elif self.number == 49:
                self.sort = sortdict[4]
                self.truenumber = 0
            else:
                print("Error: out of bounds (dat is ginne kaart maat)")
        if sort is not None:
            self.sort = sortdict[sort]

    def vocalize(self):
        if 1 < self.truenumber < 11:
            print(self.truenumber, " ", self.sort)
        elif self.truenumber == 1:
            print("aas", self.sort)
        elif self.truenumber == 11:
            print("koning", self.sort)
        elif self.truenumber == 12:
            print("koningin", self.sort)
        elif self.truenumber == 13:
            print("boer", self.sort)
        elif self.truenumber == 0:
            print("joker", self.sort)
        else:
            print("dat is geen valide kaart maatje")
    def compatible(self, _card):
        if _card.truenumber == self.truenumber:
            return True
        elif _card.sort == self.sort:
            return  True
        elif _card.truenumber == 13:
            return True
        elif _card.truenumber == 0:
            return True
        return False