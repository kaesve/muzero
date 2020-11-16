"""
Simple extension to a dictionary to access keys with a '.key' syntax instead of a ['key'] syntax.
"""
from __future__ import annotations


class DotDict(dict):

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, key, value):
        self[key] = value

    def to_json(self, file: str) -> None:
        import json
        with open(file, 'w') as f:
            json.dump(self, f)

    def recursive_update(self, other: DotDict) -> None:
        for (key, value) in other.items():
            if isinstance(value, DotDict) and isinstance(self[key], DotDict):
                self[key].recursive_update(value)
            else:
                self[key] = value

    @staticmethod
    def from_json(file: str) -> DotDict:
        import json
        with open(file, 'r') as f:
            content = json.load(f, object_hook=lambda d: DotDict(**d))
        return DotDict(content)
