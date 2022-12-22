from setuptools import setup, find_packages
from shutil import copytree
import os


# Install dependencies
setup(
    name='adaptiveXRD',
    version='0.0.1',
    description='A package designed to autonomously guide XRD measurements and identify crystalline phases.',
    author='Nathan J. Szymanski',
    author_email='nathan_szymanski@berkeley.edu',
    python_requires='>=3.6.0',
    url='https://github.com/njszym/AdaptiveXRD',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'pymatgen', 'scipy', 'scikit-image', 'tensorflow', 'pyxtal', 'pyts', 'tqdm', 'xmltodict']
)
