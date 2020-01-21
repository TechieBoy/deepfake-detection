from collections import defaultdict


class ConstDict(defaultdict):
    def __init__(self, **kwargs):
        super(ConstDict, self).__init__(ConstDict)
        for key in kwargs:
            self.__setattr__(key, kwargs[key])

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value
