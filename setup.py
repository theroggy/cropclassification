"""Setup file for the cropclassification package."""

import setuptools

with open("README.md") as fh:
    long_description = fh.read()

with open("cropclassification/version.txt") as file:
    version = file.readline()

setuptools.setup(
    name="cropclassification",
    version=version,
    author="Pieter Roggemans",
    author_email="pieter.roggemans@gmail.com",
    description="Package to classify crops based on sentinel images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theroggy/cropclassification",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "dask-core",
        "exactextract",
        "geofileops",
        "geopandas",
        "numpy",
        "openeo",
        "psutil",
        "rasterio",
        "rasterstats",
        "rioxarray",
        "scikit-learn",
        "tensorflow",
        "xlsxwriter",
    ],
    entry_points="""
        [console_scripts]
        cropclassification=cropclassification.taskrunner:main
        """,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
