from setuptools import setup, find_packages

setup(
    name ='conmo',
    version = '1.0.0',
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