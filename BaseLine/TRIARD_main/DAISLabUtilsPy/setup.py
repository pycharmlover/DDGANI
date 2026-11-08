import setuptools

setuptools.setup(
    name = "daislabutils",
    version = "0.0.1",
    author = "Tullio Pizzuti",
    author_email = "tpizzuti@unisa.it",
    description = "",
    long_description = "",
    long_description_content_type = "text/markdown",
    url = "https://github.com/tulliopizzuti/DAISLabUtilsPy",
    project_urls = {

    },
    install_requires=[
        "pyroaring",
        "pysettrie"
                      ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages = [
        "dependencies_util",
        "hashablebitmap",
        "lattice"
    ],
    python_requires = ">=3.10"
)