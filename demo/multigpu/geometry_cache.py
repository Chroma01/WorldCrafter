"""Geometry lives only within one session/chunk/pyramid stage."""


class GeometryCache:
    def __init__(self):
        self.reset()

    def reset(self):
        self.scope = None
        self.entries = {}
        self.hits = self.misses = 0

    def begin(self, scope):
        if scope != self.scope:
            self.entries.clear()
            self.scope = scope

    def get(self, key, factory):
        if self.scope is None:
            raise RuntimeError('Geometry cache requires a sampling scope')
        if key not in self.entries:
            self.misses += 1
            self.entries[key] = factory()
        else:
            self.hits += 1
        return self.entries[key]

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, entries=len(self.entries))
