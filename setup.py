import os
import glob

os.environ['DISTUTILS_DEBUG'] = "1"

from setuptools import setup, Extension
from setuptools.command import install as _install

setup(name='LBH_to_eflux',
      version = "1.0.0",
      description = "Neural Networking Modeling of Electron Total Energy Flux using LBH emission data from SSUSI",
      author = "Jason Li",
      author_email = 'jali7001@colorado.edu',
      download_url = "https://github.com/jali7001/LBH_to_eflux",
      long_description = """
                        This is the code used to build my neural network model predicting electron total energy flux from SSUSI SDR data.
            """,
      install_requires=['numpy','matplotlib','scipy','scikit-learn','apexpy',
                        'pytest','h5py','requests','netcdf4','logbook'],
      packages=['LBHL_to_eflux'],
      package_dir={'LBH_to_eflux' : 'LBH_to_eflux'},
      license='LICENSE.txt',
      zip_safe = False,
      classifiers = [
            "Development Status :: 4 - Beta",
            "Topic :: Scientific/Engineering",
            "Intended Audience :: Science/Research",
            "Natural Language :: English",
            "Programming Language :: Python"
            ],
      )
