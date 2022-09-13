"""
初始化时保存的路径
"""

__updated__ = "2022-08-18 03:06:52"

import os
import pickle


class HomePlace(object):
    """
    ```
    daily_data_file: 日频数据存放位置
    factor_data_file: （辅助、初级）因子数据存放位置
    barra_data_file: 十种常用风格因子的存放位置
    update_data_file: 更新辅助数据的存放位置
    api_token: dcube的api
    final_factor_file: 最终因子数据的存放位置
    daily_enddate: 日频数据更新截止日期
    ```
    """

    __slots__ = [
        "daily_data_file",
        "minute_data_file",
        "factor_data_file",
        "barra_data_file",
        "update_data_file",
        "api_token",
        "final_factor_file",
        "daily_enddate",
        "minute_enddate",
    ]

    def __init__(self):
        user_file = os.path.expanduser("~") + "/"
        path_file = open(user_file + "paths.settings", "rb")
        paths = pickle.load(path_file)
        for k in self.__slots__:
            setattr(self, k, paths[k])
