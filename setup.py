import sys, os, glob
import setuptools

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.build_py import build_py

class my_build_py(build_py):

    def run(self, *args, **kwargs):
        # regular build done by parent class
        build_py.run(self, *args, **kwargs)
        if not self.dry_run:  # compatibility with the parent options
            self.copy_pkg(os.path.join('numpy_benchmarks', 'benchmarks'))

    def copy_pkg(self, pkg):
        import shutil
        target = os.path.join(self.build_lib, pkg)
        shutil.rmtree(target, True)
        shutil.copytree(pkg, target)

xtl_version = '0.6.18'
xsimd_version = '7.4.8'
xtensor_version = '0.21.5'
pybind11_version = '2.5.0'
xtensor_python_version = '0.24.1'
xtensor_blas_version = '0.17.2'

class my_install(install):

    def run(self):
        install.run(self)
        self.post_install()

    def post_install(self):
        import tempfile
        with tempfile.TemporaryDirectory() as self.tmpdir:
            xtl = self.download_src("https://github.com/QuantStack/xtl/archive/{}.zip".format(xtl_version))
            self.cmake(xtl)

            xsimd = self.download_src("https://github.com/QuantStack/xsimd/archive/{}.zip".format(xsimd_version))
            self.cmake(xsimd)

            xtensor = self.download_src("https://github.com/QuantStack/xtensor/archive/{}.zip".format(xtensor_version))
            self.cmake(xtensor)

            pybind11 = self.download_src("https://github.com/pybind/pybind11/archive/v{}.zip".format(pybind11_version))
            self.cmake(pybind11,
                      '-DPYTHON_EXECUTABLE={}'.format(sys.executable))

            xpython = self.download_src("https://github.com/QuantStack/xtensor-python/archive/{}.zip".format(xtensor_python_version))
            self.cmake(xpython,
                      '-DPYTHON_EXECUTABLE={}'.format(sys.executable))

            xblas = self.download_src("https://github.com/QuantStack/xtensor-blas/archive/{}.zip".format(xtensor_blas_version))
            self.cmake(xblas)

    def download_src(self, url):
        from io import BytesIO
        from zipfile import ZipFile
        from urllib.request import urlopen
        resp = urlopen(url)
        zipfile = ZipFile(BytesIO(resp.read()))
        target_dir = os.path.join(self.tmpdir, "download")
        zipfile.extractall(target_dir)
        return os.path.join(target_dir,
                            zipfile.namelist()[0]).rstrip(os.path.sep)

    def cmake(self, target, *args):
        import subprocess
        CMAKE_BIN_DIR = subprocess.check_output(
            [sys.executable,
             '-c',
             'from cmake import CMAKE_BIN_DIR; print(CMAKE_BIN_DIR)'])
        btarget = os.path.basename(target)
        CMAKE_BIN_DIR = CMAKE_BIN_DIR.decode().strip()
        cmake = os.path.join(CMAKE_BIN_DIR, "cmake")
        build_dir = os.path.join(self.tmpdir, "build", btarget)
        os.makedirs(build_dir, exist_ok=True)
        subprocess.check_call([cmake, os.path.join("..", "..", "download",
                                                     btarget),
                               "-DCMAKE_INSTALL_PREFIX={}".format(self.prefix),
                               *args],
                              cwd=build_dir)
        subprocess.check_call([cmake, "--build", "."], cwd=build_dir)
        subprocess.check_call([cmake, "--build", ".", "--target", "install"], cwd=build_dir)


setup(
    name='numpy_benchmarks',
    version='0.1.0',
    description='A collection of numpy kernels for benchmarking',
    author='serge-sans-paille',
    author_email='serge.guelton@telecom-bretagne.eu',
    url='https://github.com/serge-sans-paille/numpy-benchmarks',
    license="BSD 3-Clause",
    install_requires=open('requirements.txt').read().splitlines(),
    packages=['numpy_benchmarks', 'numpy_benchmarks/benchmarks'],
    package_data={'numpy_benchmarks/benchmarks':
                  ['numpy_benchmarks/benchmarks/*.cpp']},
    scripts=['numpy_benchmarks/np-bench'],
    cmdclass={'install': my_install, 'build_py': my_build_py},
)
