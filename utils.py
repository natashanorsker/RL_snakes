import collections


class defaultdict2(collections.defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError((key,))

        if isinstance(self.default_factory, types.FunctionType):
            nargs = len(inspect.getfullargspec(self.default_factory).args)
            self[key] = value = self.default_factory(key) if nargs == 1 else self.default_factory()
            return value
        else:
            return super().__missing__(key)