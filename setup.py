import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("cropclassification/version.txt", mode="r") as file:
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
        "geofileops",
        "geopandas",
        "openeo",
        "psutil",
        "rasterio",
        "rasterstats",
        "scikit-learn",
        "tensorflow",
    ],
    entry_points="""
        [console_scripts]
        cropclassification=cropclassification.cropclassification:main
        """,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
