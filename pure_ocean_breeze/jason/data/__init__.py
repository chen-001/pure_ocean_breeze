"""
数据相关，包括读取本地常用的数据，向本地数据库写入数据，以及一些不常见类型文件的读取函数
"""

__all__ = ["read_data", "dicts", "tools", "database"]

from pure_ocean_breeze.jason.data import read_data
from pure_ocean_breeze.jason.data import tools
from pure_ocean_breeze.jason.data import dicts
from pure_ocean_breeze.jason.data import database
