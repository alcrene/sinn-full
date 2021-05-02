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
