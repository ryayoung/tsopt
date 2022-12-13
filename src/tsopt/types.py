

class ListData:

    def __init__(self, mod, data=None):
        self.mod = mod
        if not data:
            data = self.default_template
        super().__init__(data)

    @property
    def default_template(self):
        return []

    def _repr_html_(self):
        return "".join([item._repr_html() for item in self])
