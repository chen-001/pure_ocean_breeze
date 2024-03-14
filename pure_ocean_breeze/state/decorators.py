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


class params_setter(object):
    """用于标注设置参数部分的装饰器"""

    def __init__(self, slogan=None):
        if not slogan:
            slogan = "这是设置参数类型的函数\n"
        self.slogan = slogan
        self.box = {}

    def __call__(self, func):
        # func.__doc__=self.slogan+func.__doc__
        self.box[func.__name__] = func
        self.func = func

        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            # if not STATES['NO_LOG:
            #     logger.info(f'{func.__name__} has been called $ kind of params_setter')

        return wrapper


class main_process(object):
    """用于标记主逻辑过程的装饰器"""

    def __init__(self, slogan=None):
        if not slogan:
            slogan = "这是主逻辑过程的函数\n"
        self.slogan = slogan
        self.box = {}

    def __call__(self, func):
        # func.__doc__=self.slogan+func.__doc__
        self.box[func.__name__] = func

        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            # if STATES['NO_LOG:
            #     logger.success(f'{func.__name__} has been called $ kind of main_process')

        return wrapper


class tool_box(object):
    """用于标注工具箱部分的装饰器"""

    def __init__(self, slogan=None):
        if not slogan:
            slogan = "这是工具箱的函数\n"
        self.slogan = slogan
        self.box = {}

    def __call__(self, func):
        # func.__doc__=self.slogan+func.__doc__
        self.box[func.__name__] = func

        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            # logger.success(f'{func.__name__} has been called $ kind of tool_box')
            return res

        return wrapper


class history_remain(object):
    """用于历史遗留部分的装饰器"""

    def __init__(self, slogan=None):
        if not slogan:
            slogan = "这是历史遗留的函数\n"
        self.slogan = slogan
        self.box = {}

    def __call__(self, func):
        # func.__doc__=self.slogan+func.__doc__
        self.box[func.__name__] = func

        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            # logger.success(f'{func.__name__} has been called $ kind of history_remain')

        return wrapper
