import logging

sorts = ("harten", "ruiten", "schoppen", "klaveren", "special")


class Card:
    def __init__(self, number=None, sort=None, trueNumber=None):
        self.truenumber = -1
        if number is not None:
            self.number = number
            if self.number < 48:
                self.sort = sorts[self.number // 12]
                self.truenumber = self.number % 12 + 1
            elif self.number == 48:
                self.sort = sorts[4]
                self.truenumber = 13
            elif self.number == 49:
                self.sort = sorts[4]
                self.truenumber = 0
            else:
                print("Error: out of bounds (dat is ginne kaart maat)")
            return
        if sort is not None:
            self.sort = sort
        if trueNumber is not None:
            self.truenumber = trueNumber
            if 0 < self.truenumber < 13:
                self.number = self.truenumber * (sorts.index(self.sort) + 1)
            elif self.truenumber == 0:
                self.number = 49
            elif self.truenumber == 13:
                self.number = 48

    def __eq__(self, other):
        """
        custom == operator for card objects
        :param other: other card
        :return: bool if numbers are the same
        """
        if other.number == self.number:
            return True
        return False
    
    def __hash__(self):
        return hash(self.number)
    def toString(self):
        if 1 < self.truenumber < 11:
            return str(self.truenumber) + " " + self.sort
        elif self.truenumber == 1:
            return "aas" + " " + self.sort
        elif self.truenumber == 11:
            return "koning" + " " + self.sort
        elif self.truenumber == 12:
            return "koningin" + " " + self.sort
        elif self.truenumber == 13:
            return "boer" + " " + self.sort
        elif self.truenumber == 0:
            return "joker" + " " + self.sort
        else:
            return "dat is geen valide kaart maatje. Nummer:" + " " + str(self.truenumber)

    def vocalize(self):
        print(self.toString())

    def compatible(self, sort, truenumber):
        if truenumber == self.truenumber or self.truenumber == 0 or self.truenumber == 13 or sort == self.sort:
            return True
        # logging.debug(str(truenumber) + " and " + str(self.truenumber) + " not compatible and " + sort + " and " + self.sort + " not compatible")
        return False
