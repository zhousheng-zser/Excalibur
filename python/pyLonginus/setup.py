from distutils.core import setup, Extension, DEBUG

sfc_module = Extension('pyLonginus', sources = ['pyLonginus.cpp'])

setup(name = 'pyLonginus', version = '1.0',
    description = 'Python Package with Longinus C++ extension',
    ext_modules = [sfc_module]
    )