import dataclasses
import inspect
from numbers import Number
from npstructures import RaggedArray, RaggedShape
from itertools import accumulate
import numpy as np


def shallow_tuple(obj):
    return tuple(
        getattr(obj, field.name) for field in dataclasses.fields(obj)
    )


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


class NpDataClass:

    @classmethod
    def single_entry(cls, *args, **kwargs):
        obj = cls._single_entry(*args, **kwargs)
        cls._implicit_format_conversion(obj)
        return obj

    @classmethod
    def _implicit_format_conversion(cls, obj):
        for field in dataclasses.fields(obj):
            setattr(obj, field.name, np.asanyarray(getattr(obj, field.name)))

    def _assert_same_lens(self):
        t = shallow_tuple(self)
        l = len(t[0])
        for p in t:
            assert len(p) == l, f"All fields in a npdataclass need to be of the same length: {[len(p) for p in t]}"

    @classmethod
    def empty(cls):
        def empty_element(t):
            if t in (int, float):
                return np.empty(0, dtype=t)
            elif inspect.isclass(t) and issubclass(t, NpDataClass):
                return t.empty()
            else:
                return []
        values = (empty_element(field.type)
                  for field in dataclasses.fields(cls))
        return cls(*values)

    def astype(self, new_class):
        my_fields = {f.name for f in dataclasses.fields(self)}
        new_fields = {f.name for f in dataclasses.fields(new_class)}
        assert all(
            field.name in my_fields for field in dataclasses.fields(new_class)
        ), (my_fields, new_fields)
        return new_class(**{name: getattr(self, name) for name in new_fields})

    def shallow_tuple(self):
        return tuple(
            getattr(self, field.name) for field in dataclasses.fields(self)
        )

    def __getitem__(self, idx):
        cls = self.single_entry if isinstance(idx, Number) else self.__class__
        fields = [f[idx] for f in shallow_tuple(self)]

        return cls(*fields)

    def __len__(self):
        return len(shallow_tuple(self)[0])

    def __array_function__(self, func, types, args, kwargs):
        if func == np.concatenate:
            objects = args[0]
            tuples = [shallow_tuple(o) for o in objects]
            columns = []
            for t in zip(*tuples):
                # Removed, a bit too strict
                #assert all(id(type(t[0])) == id(type(i)) for i in t), ([type(v) for v in t], [id(type(v)) for v in t])
                columns.append(np.concatenate(list(t)))
            return self.__class__(*columns)
            # np.concatenate(list(t))
            #                    for t in zip(*tuples)))
        if func == np.equal:
            one, other = args
            return all(
                np.equal(s, o)
                for s, o in zip(shallow_tuple(one), shallow_tuple(other))
            )

        return NotImplemented

    def __iter__(self):
        return (self[i] for i in range(len(self)))
                # return (self.single_entry(*comb) for comb in zip(*shallow_tuple(self)))


    @classmethod
    def stack_with_ragged(cls, objects):
        tuples = [shallow_tuple(o) for o in objects]
        new_entries = [list(t) for t in zip(*tuples)]
        new_entries = (
            RaggedArray(np.concatenate(e), RaggedShape([len(t) for t in e]))
            if hasattr(e[0], "__len__") and not all(len(i) == len(e[0]) for i in e)
            else np.concatenate(e).reshape(-1, len(e[0]))
            for e in new_entries
        )
        ret = cls(*new_entries)
        return ret


def npdataclass(base_class):
    new_class = dataclasses.dataclass(base_class)

    class FinalClass(new_class, NpDataClass):
        _single_entry = dataclasses.make_dataclass(base_class.__name__,
                                                   [(f.name, f.type, f) for f in dataclasses.fields(new_class)])
        dataclass = _single_entry



        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._implicit_format_conversion(self)
            self._assert_same_lens()

        def __eq__(self, other):
            for s, o in zip(shallow_tuple(self), shallow_tuple(other)):
                if not s.shape == o.shape:
                    return False
                if not np.all(np.equal(s, o)):
                    return False
            return True
            # return all(np.all(np.equal(s, o)) for s, o in zip(self.shallow_tuple(), other.shallow_tuple()))

        def __str__(self):
            lines = []
            col_length = 25
            lines.append(f"{self.__class__.__name__} with {len(self)} entries")
            header = []
            field_names = [field.name for field in dataclasses.fields(self)]
            for name in field_names:
                header.append(f"{name:>{col_length}}")
            lines.append("".join(header))
            for _, entry in zip(range(10), self):
                cols = [getattr(entry, name) for name in field_names]
                lines.append("".join(f"{str(col)[:col_length-2]:>{col_length}}" for col in cols))
            return "\n".join(lines)


        __repr__ = __str__



    FinalClass.__name__ = base_class.__name__
    FinalClass.__qualname__ = base_class.__qualname__
    return FinalClass
