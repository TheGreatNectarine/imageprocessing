import sys


class ConstError(AttributeError):
    pass


class Const:
    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise ConstError(f"Cannot rebind constant value \"{key}\"")
        self.__dict__[key] = value

    def __getattr__(self, item):
        return self.__dict__[item]


sys.modules[__name__] = Const()
