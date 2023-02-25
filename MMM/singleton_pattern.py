"""
* Decorate a class with '@Singleton' to make it a Singleton

To dos:
"""


class Singleton:

    def __init__(self, decorated):
        self.__decorated = decorated

    def instance(self):
        try:
            return self.__instance
        except AttributeError:
            self.__instance = self.__decorated()
            return self.__instance

    def __call__(self):
        raise TypeError('[!] Singletons must be accessed through `instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self.__decorated)
