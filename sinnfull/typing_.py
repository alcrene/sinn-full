import mackelab_toolbox.typing as mtbtyping

class IndexableNamespace(mtbtyping.IndexableNamespace):
    def __getattr__(self, attr):
        if hasattr(mtbtyping.IndexableNamespace, '__getattr__'):
            try:
                return super().__getattr__(attr)
            except AttributeError:
                pass
        if attr != '_tags' and attr in getattr(self, '_tags', set()):
            return self
        else:
            raise AttributeError(
                f"'{attr}' was not found in the IndexableNamespace")

    @classmethod
    def json_encoder(cls, idxns):
        return {k:v for k,v in idxns.__dict__.items()
                if k != '_tags' and k != '_selfrefs'}

    def get_subparams(submodel):
        subparams = {k.split('.', 1)[1]: v for k,v in self
                     if k.split('.')[0] == submodel}
        if subparams:
            return IndexableNamespace(**subparams)
        else:
            raise AttributeError("IndexableNamespace has no parameters "
                                 f"associated with a submodel '{submodel}'.")
