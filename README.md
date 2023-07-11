# pygnme

The `pygnme` package is a python interface to the [`libgnme`](https://github.com/hgaburton/libgnme) package.

### Prequisites

`pygnme` requires the pybind11 package, which can be installed with 

```
python -m pip install pybind11
```

### Installation

`pygnme` is packaged with a `setup.py`:

```
git clone --recurse-submodules git@github.com:hgaburton/pygnme.git
python -m pip install . -v --user
```

If an error such as `ImportError: libgnme_wick.so: cannot open shared object file: No such file or directory` is encountered, add the `site-packages` directory corresponding to your python instance to `LD_LIBRARY_PATH`.
