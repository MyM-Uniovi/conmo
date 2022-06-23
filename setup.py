from setuptools import setup, find_packages

# Read the contents of README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name ='conmo',
    version = '1.0.1',
    author = 'GrupoMyM',
    author_email = 'mym.inv.uniovi@gmail.com',
    url ='https://github.com/MyM-Uniovi/conmo',
    maintainer ='GrupoMyM',
    classifiers = [
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
    ],
    license_files = 'GNU GPLv3',
    long_description = long_description, 
    long_description_content_type = 'text/markdown',
    description = 'Conmo is a framework developed in Python whose main objective \
            is to facilitate the execution and comparison of different anomaly detection and  experiments.',
    keywords = [
        'conmo',
        'machine learning',
        'time series',
        'anomaly detection',
        'outlier detection',
        'condition monitoring'
    ],
    packages = find_packages(),
    install_requires = [
        'numpy',
        'pandas',
        'tensorflow',
        'requests',
        'scipy',
        'scikit-learn',
        'pyarrow'
    ],
    python_requires = '>=3.7,<3.10'
    )