"""
用于标注函数功能的一些装饰器（用处不大）
"""

__updated__ = "2023-08-15 17:03:08"
from typing import Iterable


def _list_value(x, list_num_order):
    if isinstance(x, Iterable):
        return x[list_num_order]
    else:
        return x


def _dict_value(x, list_num_order):
    dfs = {}
    for k, v in x.items():
        if isinstance(v, Iterable):
            dfs[k] = v[list_num_order]
        else:
            dfs[k] = v
    return dfs


def do_on_dfs(func):
    def wrapper(df=None, *args, **kwargs):
        if isinstance(df, list) or isinstance(df,tuple):
            dfs = [
                func(
                    i, *[_list_value(i, num) for i in args], **_dict_value(kwargs, num)
                )
                for num, i in enumerate(df)
            ]
            return dfs
        else:
            return func(df, *args, **kwargs)

    return wrapper



