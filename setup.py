from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as f:
        return f.read().splitlines()

setup(
    name='telco_churn_library',  # Library name
    version='0.1',  # Version number
    packages=find_packages(),  # Automatically find all packages
    install_requires=parse_requirements('requirements.txt'),  # Read from requirements.txt
    description='Churn Rate Prediction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Deepak, Tarang and Enzo from DSDM 2025',
    author_email='tarang.kadyan@bse.eu',
    url='https://github.com/Tarangkd/telco_churn_library.git',  # URL of our project repository
    classifiers=[  # Classifiers for Python Package Index (PyPI)
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum Python version
)
