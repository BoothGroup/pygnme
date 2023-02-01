import os
import sys
import pathlib

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        tmp = pathlib.Path(self.build_temp)
        tmp.mkdir(parents=True, exist_ok=True)

        ext = pathlib.Path(self.get_ext_fullpath(ext.name))
        ext.mkdir(parents=True, exist_ok=True)

        config = "Debug" if self.debug else "Release"
        cmake_args = [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(ext.parent.absolute()),
                "-DCMAKE_BUILD_TYPE=" + config,
                "-DCMAKE_INSTALL_RPATH=@ORIGIN",
                "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]
        build_args = ["-j4"]

        os.chdir(str(tmp))
        self.spawn(["cmake", str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", "."] + build_args)
        os.chdir(str(cwd))


setup(
        name="pygnme",
        version="0.0.0",
        packages=["pygnme"],
        ext_modules=[CMakeExtension(".")],
        cmdclass={
            "build_ext": build_ext,
        },
)
