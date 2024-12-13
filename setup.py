from setuptools import setup, find_packages

# Read dependencies from the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Configuration of setup.py
setup(
    name="telco-churn-library",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    description="Library for analyzing datasets",
    author="Tarang Kadyan, Deepak Malik, Enzo Infantes",
    author_email="deepak.malik@bse.eu, tarang.kadyan@bse.eu, enzo.infantes@bse.eu",
    url='https://github.com/Tarangkd/telco_churn_library.git',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

