from setuptools import setup, find_packages

setup(
    name='cogtasks_rnn_modularity',
    version='1.0.0',
    url='https://github.com/alexnegron/cogtasks_rnn_modularity.git',
    author='Alex Negron',
    author_email='negrona@mit.edu',
    description='Code for modularity project',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)