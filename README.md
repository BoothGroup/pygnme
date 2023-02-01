# pygnme

The `pygnme` package is a python interface to the [`libgnme`](https://github.com/hgaburton/libgnme) package.

### Installation

`pygnme` is packaged with a `setup.py`:

```
git clone https://github.com/BoothGroup/pygnme
python -m pip install . -v --user
```

If an error such as `ImportError: libgnme_wick.so: cannot open shared object file: No such file or directory` is encountered, add the `site-packages` directory corresponding to your python instance to `LD_LIBRARY_PATH`.
