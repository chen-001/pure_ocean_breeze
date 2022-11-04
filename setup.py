__updated__ = "2022-11-05 00:06:43"

from setuptools import setup
import setuptools
import re
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def get_version(package):
    """Return package version as listed in `__version__` in `init.py`."""
    init_py = open(os.path.join(package, "__init__.py")).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


setup(
    name="pure_ocean_breeze",
    version=get_version("pure_ocean_breeze"),
    description="众人的回测框架",
    # long_description="详见homepage\nhttps://github.com/chen-001/pure_ocean_breeze.git",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="chenzongwei",
    author_email="winterwinter999@163.com",
    url="https://github.com/chen-001/pure_ocean_breeze.git",
    project_urls={"Documentation": "https://chen-001.github.io/pure_ocean_breeze/"},
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "plotly",
        "matplotlib",
        "pyarrow",
        "loguru",
        "cachier",
        "knockknock",
        "dcube",
        "tenacity",
        "alphalens",
        "pickledb",
        "pymysql",
        "sqlalchemy",
        "SciencePlots",
        "psycopg2",
        "requests",
        "bs4",
        "wrapt_timeout_decorator",
        "pyfinance",
        "texttable",
        "numpy_ext",
        "xpinyin",
        "cufflinks",
        "clickhouse_sqlalchemy",
    ],
    python_requires=">=3",
    license="MIT",
    packages=setuptools.find_packages(),
    requires=[],
)
