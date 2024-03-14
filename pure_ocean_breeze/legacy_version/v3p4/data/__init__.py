"""
数据相关，包括读取本地常用的数据，向本地数据库写入数据，以及一些不常见类型文件的读取函数
"""

__all__ = ["read_data", "write_data", "tools", "dicts", "database"]

from pure_ocean_breeze.legacy_version.v3p4.data import read_data
from pure_ocean_breeze.legacy_version.v3p4.data import tools
from pure_ocean_breeze.legacy_version.v3p4.data import dicts
from pure_ocean_breeze.legacy_version.v3p4.data import database
from pure_ocean_breeze.legacy_version.v3p4.data import write_data
