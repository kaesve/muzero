"""
Simple extension to a dictionary to access keys with a '.key' syntax instead of a ['key'] syntax.
"""
from __future__ import annotations
import typing


class DotDict(dict):

    def __getattr__(self, name: typing.Generic) -> typing.Generic:
        return self[name]

    def __setattr__(self, key: typing.Generic, value: typing.Generic) -> None:
        self[key] = value

    def copy(self) -> DotDict:
        new = DotDict()
        for k, v in self.items():
            new[k] = v.copy() if isinstance(v, DotDict) else v
        return new

    def to_json(self, file: str) -> None:
        import json
        with open(file, 'w') as f:
            json.dump(self, f)

    def recursive_update(self, other: DotDict) -> None:
        for k, v in other.items():
            if isinstance(v, DotDict) and isinstance(self[k], DotDict):
                self[k].recursive_update(v)
            else:
                self[k] = v

    @staticmethod
    def from_json(file: str) -> DotDict:
        import json
        with open(file, 'r') as f:
            content = json.load(f, object_hook=lambda d: DotDict(**d))
        return DotDict(content)
