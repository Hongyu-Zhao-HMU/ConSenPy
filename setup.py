from setuptools import setup, find_packages

setup(
    name="ConSenPy",
    version="1.0.0",
    packages=["ConSenPy"],
    include_package_data=True,
    package_data={"": ["data/*.csv"],},
    install_requires=[
        "pyro-ppl==1.9.1",
        "numpy==1.23.5",
        "pandas==2.3.3",
        "scanpy==1.10.3",
        "anndata==0.10.9",
        "scipy==1.11.4",
        "scikit-learn==1.6.1",
        "statsmodels==0.14.1",
    ],
    author="Hongyu Zhao",
    description="ConSenPy: detecting context-specific senescence genes from single-cell and spatial transcriptomics",
)