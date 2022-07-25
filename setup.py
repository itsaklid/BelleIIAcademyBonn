from setuptools import setup, find_packages
from pkg_resources import resource_filename
from pathlib import Path


with (Path(__file__).parent / 'readme.md').open() as readme_file:
    readme = readme_file.read()

setup(
    name='BelleIIAcademyBonn',
    packages=find_packages(),
    url="",
    author='Markus Tobias Prim',
    author_email='markus.prim@cern.ch',
    description='''
Software for the Belle II Academy in Bonn.
''',
    install_requires=[
        'numpy<1.23',  # Remove this when numba becomes compatible with numpy>=1.23
        'scipy',
        'numba',
        'matplotlib', 
        'jupyterlab', 
        'uncertainties',
        'iminuit',
        'gvar',
        'numdifftools',
    ],
    extras_require={
        "examples":  [],
    },
    long_description=readme,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License "
    ],
    license='MIT',
)
