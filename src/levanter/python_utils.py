# https://stackoverflow.com/questions/62881486/a-group-of-constants-in-python
class StringHolderEnum(type):
    """Like a python enum but just holds string constants, as opposed to wrapped string constants"""

    def __new__(cls, name, bases, members):
        # this just iterates through the class dict and removes
        # all the dunder methods
        cls.members = [v for k, v in members.items() if not k.startswith("__") and not callable(v)]
        return super().__new__(cls, name, bases, members)

    # giving your class an __iter__ method gives you membership checking
    # and the ability to easily convert to another iterable
    def __iter__(cls):
        yield from cls.members
