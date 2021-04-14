from rosnet.utils import isunique


class Signature(object):
    def __init__(self, sign):
        assert all(len(i) == 1 for i in sign)
        assert len(set(sign)) == len(sign)
        self._sign = sign

    @property
    def sign(self):
        return self._sign

    def intersection(self, other):
        return Signature(filter(lambda x: x in other.sign, self.sign))

    def difference(self, other):
        return Signature(filter(lambda x: x not in other.sign, self.sign))

    def isperm(self, other) -> bool:
        if isinstance(other, Signature):
            return set(self.sign) == set(other.sign)
        elif isinstance(other, List):
            if isinstance(other[0], str):
                return set(self.sign) == set(other)
            # elif isinstance(other[0], int):
            #     return len(set(other)) == len(other) and max(other) == len(self.sign) - 1 and min(other) == 0
            raise TypeError(
                "Invalid type for is_permutation using list: %s" % str(type(other[0])))
        elif isinstance(other, str):
            return set(self.sign) == set(list(other))

        raise TypeError("Invalid type for is_permutation: %s" %
                        str(type(other)))

    def perm(self, other):
        if not self.isperm(other):
            raise ValueError("Invalid permutation")

        # TODO decompose in match blocks
        matches = []
        match = ''
        for j in other.sign:
            if match == '':
                match = j
                continue  # already know that j is in a and b
            if match + j in self.sign:
                match.append(j)
            else:
                matches.append(match)
                match = ''
        if match != '':
            matches.append(match)

    def __str__(self):
        return "".join(self.sign)

    def __getitem__(self, key):
        # return index character if number passed
        if isinstance(key, int):
            return self.sign[key]

        # return index number if character passed
        elif isinstance(key, str):
            assert len(key) == 1
            return next(i for (i, v) in enumerate(self.sign) if v == key)

        raise TypeError("Invalid type for indexing")

    def __setitem__(self, key, value):
        """ Swap indexes in 'key'(int) or 'key'(str) to index in 'value'(int) or 'value'(str).
        """
        if not isinstance(key, str) or not isinstance(key, int):
            raise TypeError("Invalid type for indexing: %s" % str(type(key)))
        if not isinstance(value, str) or not isinstance(value, int):
            raise TypeError("Invalid type for setting: %s" % str(type(value)))
        if isinstance(key, int):
            assert 0 <= key < len(self)
        if isinstance(value, int):
            assert 0 <= key < len(self)
        if isinstance(value, str):
            assert value.isalpha() and len(value) == 1

        key_loc = self[key] if isinstance(key, str) else key
        value_loc = self[value] if isinstance(value, str) else value
        self.sign[key_loc], self.sign[value_loc] = self.sign[value_loc], self.sign[key_loc]

    def __len__(self):
        return len(self._sign)

    def __eq__(self, other) -> bool:
        return all(x == y for x, y in zip(self.sign, other.sign))

    def __sub__(self, other):
        if isinstance(other, str):
            assert len(other) == 1
            self._sign.remove(other)

        elif isinstance(other, int):
            del self._sign[other]

        elif isinstance(other, Signature):
            return self.difference(other)

        raise ValueError("Invalid type for substraction: %s" %
                         str(type(other)))
