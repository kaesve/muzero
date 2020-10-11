"""
Simple extension to a dictionary to access keys with a '.key' syntax instead of a ['key'] syntax.
"""
from __future__ import annotations


class DotDict(dict):

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, key, value):
        self[key] = value

    @staticmethod
    def from_json(file: str) -> DotDict:
        import json
        with open(file) as f:
            content = json.load(f, object_hook=lambda d: DotDict(**d))
        return DotDict(content)
