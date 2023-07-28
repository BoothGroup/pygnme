# pygnme

The `pygnme` package is a python interface to the [`libgnme`](https://github.com/hgaburton/libgnme) package. 

### Prequisites

`pygnme` requires the pybind11 package, which can be installed with 

```
python -m pip install pybind11
```
and set `pybind11_DIR` environment variable to `[python site-packages directory]/pybind11/share/cmake/`.
### Installation

`pygnme` is packaged with a `setup.py`:

```
git clone --recurse-submodules git@github.com:hgaburton/pygnme.git
python -m pip install . -v --user
```

If an error such as `ImportError: libgnme_wick.so: cannot open shared object file: No such file or directory` is encountered, add the `site-packages` directory corresponding to your python instance to `LD_LIBRARY_PATH`. 
If `cblas` related errors are encountered during compilation, add the following lines to `external/libgnme/CMakeLists.txt`
```
find_package(CBLAS)
```
and the following lines to `external/libgnme/wick/CMakeLists.txt`
```
target_link_libraries(gnme_wick gnme_utils "${BLAS_LIBRARIES}" )
include_directories("${CBLAS_LIBRARIES}")
```
