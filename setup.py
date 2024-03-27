import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):

  def __init__(self, name: str, sourcedir: str = "") -> None:
    super().__init__(name, sources=[])
    self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):

  def build_extension(self, ext: CMakeExtension) -> None:
    # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
    ext_fullpath = Path.cwd() / self.get_ext_fullpath(
        ext.name)  # type: ignore[no-untyped-call]
    extdir = ext_fullpath.parent.resolve()

    # Using this requires trailing slash for auto-detection & inclusion of
    # auxiliary "native" libs

    debug = self.debug
    if self.debug is None:
        debug = int(os.environ.get("DEBUG", 0))
    cfg = "Debug" if debug else "Release"

    # CMake lets you override the generator - we need to check this.
    # Can be set with Conda-Build, for example.
    cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

    # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
    # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
    # from Python.
    cmake_args = [
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        f"-DTORCH_EXTENSION_NAME={ext.name.split('.')[-1]}"
    ]
    build_args = []
    # Adding CMake arguments set as environment variable
    # (needed e.g. to build for ARM OSx on conda-forge)
    if "CMAKE_ARGS" in os.environ:
      cmake_args += [
          item for item in os.environ["CMAKE_ARGS"].split(" ") if item
      ]

    # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
    # across all generators.
    if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
      # self.parallel is a Python 3 only way to set parallel jobs by hand
      # using -j in the build_ext call, not supported by pip or PyPA-build.
      if hasattr(self, "parallel") and self.parallel:
        # CMake 3.12+ only.
        build_args += [f"-j{self.parallel}"]

    build_temp = Path(self.build_temp) / ext.name
    if not build_temp.exists():
      build_temp.mkdir(parents=True)

    subprocess.run(["cmake", ext.sourcedir] + cmake_args,
                   cwd=build_temp,
                   check=True)
    subprocess.run(["cmake", "--build", "."] + build_args,
                   cwd=build_temp,
                   check=True)


setup(ext_modules=[CMakeExtension("minuet.nn.functional._C")],
      cmdclass={'build_ext': CMakeBuild},
      zip_safe=False)
