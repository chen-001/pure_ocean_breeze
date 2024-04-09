__updated__ = "2023-07-14 15:12:11"

from setuptools import setup
import setuptools
import re
import os
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def get_version(package):
    """Return package version as listed in `__version__` in `init.py`."""
    init_py = open(os.path.join(package, "__init__.py")).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


install_requires = [
    # "numpy",
    "pandas<=1.5.3",
    "scipy",
    "statsmodels",
    "plotly",
    # "matplotlib",
    "pyarrow",
    "loguru",
    "knockknock",
    "dcube",
    "tenacity",
    "pickledb",
    "pymysql",
    "sqlalchemy",
    "requests",
    "bs4",
    "wrapt_timeout_decorator",
    "pyfinance",
    "texttable",
    "numpy_ext",
    "xpinyin",
    "cufflinks",
    "clickhouse_sqlalchemy",
    "tradetime",
    "deprecation",
    "questdb==1.1.0",
    "mpire",
    "py7zr",
    "unrar",
    "rarfile",
    # "zipfile",
    "chardet",
]
if sys.platform.startswith("win"):
    install_requires = install_requires + ["psycopg2"]
else:
    install_requires = install_requires + ["psycopg2-binary"]

setup(
    name="pure_ocean_breeze",
    version=get_version("pure_ocean_breeze"),
    description="stock factor test",
    # long_description="详见homepage\nhttps://github.com/chen-001/pure_ocean_breeze.git",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="chenzongwei",
    author_email="winterwinter999@163.com",
    url="https://github.com/chen-001/pure_ocean_breeze.git",
    project_urls={"Documentation": "https://chen-001.github.io/pure_ocean_breeze/"},
    install_requires=install_requires,
    python_requires=">=3",
    license="MIT",
    packages=setuptools.find_packages(),
    requires=[],
    extras_require={
        "windows": ["psycopg2"],
        "macos": ["psycopg2-binary"],
        "linux": ["psycopg2-binary"],
    },
)
