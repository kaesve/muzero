class DotDict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, key, value):
        self[key] = value
