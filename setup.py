import re
from setuptools import setup, find_packages
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cogtasks_rnn_modularity'))


if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

]

setup(name='cogtasks_rnn_modularity',
      install_requires=[
          'numpy',
          'matplotlib',
      ],
      author='Alex Negron')