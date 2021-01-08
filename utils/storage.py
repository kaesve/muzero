"""
Simple extension to a dictionary to access keys with a '.key' syntax instead of a ['key'] syntax.
"""
from __future__ import annotations
import typing


class DotDict(dict):
    """
    Added functionality for Python 'dict' for accessing/ setting entries as if it was a class attribute.
    Also adds functionality for json I/O and recursive updating.
    """

    def __getattr__(self, name: typing.Generic) -> typing.Generic:
        """
        Access value within a dictionary as if accessing an attribute. d[k] -> d.k
        :param name: Key attribute within the dictionary.
        :return: Value assigned to dictionary entry.
        """
        return self[name]

    def __setattr__(self, key: typing.Generic, value: typing.Generic) -> None:
        """
        Set value within a dictionary as if assigning an attribute. d[k] = v -> d.k = v
        :param key: Key attribute within the dictionary.
        :param value: Value assigned to the key entry within the dictionary.
        """
        self[key] = value

    def copy(self) -> DotDict:
        """
        Recursively copy elements within the data container.
        :return: DotDict Copy of self
        """
        new = DotDict()
        for k, v in self.items():
            new[k] = v.copy() if isinstance(v, DotDict) else v
        return new

    def to_json(self, file: str) -> None:
        """ Dump data as a JSON to the specified file """
        import json
        with open(file, 'w') as f:
            json.dump(self, f)

    def recursive_update(self, other: DotDict) -> None:
        """
        Recursively override elements within the current dictionary using the entries within the given DotDict.
        :param other: DotDict Data to override self with.
        """
        for k, v in other.items():
            if isinstance(v, DotDict) and isinstance(self[k], DotDict):
                self[k].recursive_update(v)
            else:
                self[k] = v

    @staticmethod
    def from_json(file: str) -> DotDict:
        """ Import DotDict from a JSON given by the specified file """
        import json
        with open(file, 'r') as f:
            content = json.load(f, object_hook=lambda d: DotDict(**d))  # object hook maps dict to DotDict.
        return DotDict(content)
