import dataclasses
from npstructures import RaggedArray
from itertools import accumulate
import numpy as np


class SeqArray(np.ndarray):
    @staticmethod
    def asseqarray(array):
        if isinstance(array, str):
            return np.array([ord(c) for c in array], dtype=np.uint8)
        elif isinstance(array, list):
            return np.array([SeqArray.asseqarray(row) for row in array])
        else:
            return array


class VarLenArray:
    def __init__(self, array):
        self.array = array
        self.shape = self.array.shape
        self.dtype = self.array.dtype

    def __array_function__(self, func, types, args, kwargs):
        if func == np.concatenate:
            arrays = [v.array for v in args[0]]
            lens = [len(arg) for arg in arrays]
            sizes = [arg.shape[-1] for arg in arrays]
            max_size = max(sizes)
            if all(size == max_size for size in sizes):
                return self.__class__(np.concatenate(arrays))
            ret = np.zeros_like(self.array, shape=(sum(lens), max_size))
            for end, l, a, size in zip(accumulate(lens), lens, arrays, sizes):
                ret[end - l : end, -size:] = a
            return self.__class__(ret)
        if func == np.equal:
            raise Exception()
        return NotImplemented

    def __eq__(self, other):
        return self.array == other.array

    def __repr__(self):
        return "VL" + repr(self.array)

    __str__ = __repr__

    def __neq__(self, other):
        return self.array != other.array

    def __len__(self):
        return len(self.array)

    def __array__(self, *args, **kwargs):
        return self.array

    def __iter__(self):
        return iter(self.array)

    def __getitem__(self, idx):
        return self.__class__(self.array[idx])


def npdataclass(base_class):
    new_class = dataclasses.dataclass(base_class)

    class NpDataClass(new_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for field in dataclasses.fields(self):
                if field.type == np.ndarray:
                    setattr(self, field.name, np.asanyarray(getattr(self, field.name)))
                elif field.type == SeqArray:
                    setattr(
                        self, field.name, SeqArray.asseqarray(getattr(self, field.name))
                    )

        def __str__(self):
            lines = []
            col_length = 15
            lines.append(self.__class__.__name__)
            header = []
            field_names = [field.name for field in dataclasses.fields(self)]
            for name in field_names:
                header.append(f"{name:>{col_length}}")
            lines.append("".join(header))
            for _, entry in zip(range(10), self):
                cols = [getattr(entry, name) for name in field_names]
                lines.append("".join(f"{str(col)[:col_length-2]:>{col_length}}" for col in cols))
            return "\n".join(lines)

        # def __repr__(self):
        #    fields = dataclasses.fields(self)
        #    return f"{self.__class__.__name__}({field.name}: {getattr(self, field.name)}"

        @classmethod
        def empty(cls):
            return cls(*(np.array([]) for field in dataclasses.fields(cls)))

        def astype(self, new_class):
            my_fields = {f.name for f in dataclasses.fields(self)}
            new_fields = {f.name for f in dataclasses.fields(new_class)}
            assert all(
                field.name in my_fields for field in dataclasses.fields(new_class)
            ), (my_fields, new_fields)
            return new_class(**{name: getattr(self, name) for name in new_fields})

        def __post_init__(self):
            for field in dataclasses.fields(self):
                if field.type == np.ndarray:
                    setattr(self, field.name, np.asanyarray(getattr(self, field.name)))
                elif field.type == SeqArray:
                    setattr(
                        self, field.name, SeqArray.asseqarray(getattr(self, field.name))
                    )
            if hasattr(super(), "__post_init__"):
                super().__post_init__()

        def shallow_tuple(self):
            return tuple(
                getattr(self, field.name) for field in dataclasses.fields(self)
            )

        def __getitem__(self, idx):
            return self.__class__(*[f[idx] for f in self.shallow_tuple()])

        def __len__(self):
            return len(self.shallow_tuple()[0])

        def __eq__(self, other):
            for s, o in zip(self.shallow_tuple(), other.shallow_tuple()):
                if not np.all(np.equal(s, o)):
                    return False
            return True
            # return all(np.all(np.equal(s, o)) for s, o in zip(self.shallow_tuple(), other.shallow_tuple()))

        def __array_function__(self, func, types, args, kwargs):
            if func == np.concatenate:
                objects = args[0]
                tuples = [o.shallow_tuple() for o in objects]
                return self.__class__(*(np.concatenate(list(t)) for t in zip(*tuples)))
            if func == np.equal:
                one, other = args
                return all(
                    np.equal(s, o)
                    for s, o in zip(one.shallow_tuple(), other.shallow_tuple())
                )

            return NotImplemented

        def __iter__(self):
            return (self.__class__(*comb) for comb in zip(*self.shallow_tuple()))

        @classmethod
        def stack_with_ragged(cls, objects):
            tuples = [o.shallow_tuple() for o in objects]
            new_entries = (list(t) for t in zip(*tuples))
            new_entries = (
                RaggedArray(e)
                if hasattr(e[0], "__len__") and not all(len(i) == len(e[0]) for i in e)
                else np.array(e)
                for e in new_entries
            )
            return cls(*new_entries)

    # class full_class(new_class, NpDataClass):
    #     pass
    NpDataClass.__name__ = base_class.__name__
    NpDataClass.__qualname__ = base_class.__qualname__
    return NpDataClass
