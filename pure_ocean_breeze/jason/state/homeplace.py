"""
初始化时保存的路径
"""

__updated__ = "2025-02-26 10:54:51"

import os
import pickle


class HomePlace(object):
    """
    ```
    daily_data_file: 日频数据存放位置
    factor_data_file: （辅助、初级）因子数据存放位置
    barra_data_file: 十种常用风格因子的存放位置
    ```
    """

    __slots__ = [
        "daily_data_file",
        "factor_data_file",
        "barra_data_file",
    ]

    def __init__(self):
        user_file = os.path.expanduser("~") + "/"
        path_file = open(user_file + "paths.settings", "rb")
        paths = pickle.load(path_file)
        for k in self.__slots__:
            setattr(self, k, paths[k])
