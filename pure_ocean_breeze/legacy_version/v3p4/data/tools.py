"""
针对一些不常见的文件格式，读取数据文件的一些工具函数，以及其他数据工具
"""

__updated__ = "2022-11-05 00:13:16"

import os
import pandas as pd
import tqdm
import datetime
import scipy.io as scio
import numpy as np
import numpy_ext as npext
import scipy.stats as ss
from functools import reduce, partial
from loguru import logger
from typing import Callable, Union

try:
    import rqdatac

    rqdatac.init()
except Exception:
    print("暂时未连接米筐")
from pure_ocean_breeze.legacy_version.v3p4.state.homeplace import HomePlace
from pure_ocean_breeze.legacy_version.v3p4.state.states import is_notebook


def read_h5(path: str) -> dict:
    """
    Reads a HDF5 file into a dictionary of pandas DataFrames.

    Parameters
    ----------
    path : str
        The path to the HDF5 file.

    Returns
    -------
    `dict`
        A dictionary of pandas DataFrames.
    """
    res = {}
    import h5py

    a = h5py.File(path)
    for k, v in tqdm.tqdm(list(a.items()), desc="数据加载中……"):
        value = list(v.values())[-1]
        col = [i.decode("utf-8") for i in list(list(v.values())[0])]
        ind = [i.decode("utf-8") for i in list(list(v.values())[1])]
        res[k] = pd.DataFrame(value, columns=col, index=ind)
    return res


def read_h5_new(path: str) -> pd.DataFrame:
    """读取h5文件

    Parameters
    ----------
    path : str
        h5文件路径

    Returns
    -------
    `pd.DataFrame`
        读取字典的第一个value
    """
    import h5py

    a = h5py.File(path)
    v = list(a.values())[0]
    v = a[v.name][:]
    return pd.DataFrame(v)


def read_mat(path: str) -> pd.DataFrame:
    """读取mat文件

    Parameters
    ----------
    path : str
        mat文件路径

    Returns
    -------
    `pd.DataFrame`
        字典的第4个value
    """
    return list(scio.loadmat(path).values())[3]


def convert_code(x: str) -> tuple[str, str]:
    """将米筐代码转换为wind代码，并识别其是股票还是指数

    Parameters
    ----------
    x : str
        米筐的股票/指数代码，以 XSHE 或 XSHG 结尾

    Returns
    -------
    `tuple[str,str]`
        转换后的股票/指数代码，以及该代码属于股票还是指数
    """
    x1 = x.split("/")[-1].split(".")[0]
    x2 = x.split("/")[-1].split(".")[1]
    if x2 == "XSHE":
        x2 = ".SZ"
    elif x2 == "XSHG":
        x2 = ".SH"
    x = x1 + x2
    if (x1[0] == "0" or x1[:2] == "30") and x2 == ".SZ":
        kind = "stock"
    elif x1[0] == "6" and x2 == ".SH":
        kind = "stock"
    else:
        kind = "index"
    return x, kind


def get_value(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """很多因子计算时，会一次性生成很多值，使用时只取出一个值

    Parameters
    ----------
    df : pd.DataFrame
        每个value是一个列表或元组的pd.DataFrame
    n : int
        取第n个值

    Returns
    -------
    `pd.DataFrame`
        仅有第n个值构成的pd.DataFrame
    """

    def get_value_single(x, n):
        try:
            return x[n]
        except Exception:
            return np.nan

    df = df.applymap(lambda x: get_value_single(x, n))
    return df


def indus_name(df: pd.DataFrame, col_name: str = None) -> pd.DataFrame:
    """将2021版申万行业的代码，转化为对应行业的名字

    Parameters
    ----------
    df : pd.DataFrame
        一个包含申万一级行业代码的pd.DataFrame，其中某一列或index为行业代码
    col_name : str, optional
        仅某列为行业代码时指定该参数，该列的名字，否则默认转化index, by default None

    Returns
    -------
    `pd.DataFrame`
        转化后的pd.DataFrame
    """
    names = pd.DataFrame(
        {
            "indus_we_cant_same": [
                "801170.SI",
                "801010.SI",
                "801140.SI",
                "801080.SI",
                "801780.SI",
                "801110.SI",
                "801230.SI",
                "801950.SI",
                "801180.SI",
                "801040.SI",
                "801740.SI",
                "801890.SI",
                "801770.SI",
                "801960.SI",
                "801200.SI",
                "801120.SI",
                "801710.SI",
                "801720.SI",
                "801880.SI",
                "801750.SI",
                "801050.SI",
                "801790.SI",
                "801150.SI",
                "801980.SI",
                "801030.SI",
                "801730.SI",
                "801160.SI",
                "801130.SI",
                "801210.SI",
                "801970.SI",
                "801760.SI",
            ],
            "行业名称": [
                "交通运输",
                "农林牧渔",
                "轻工制造",
                "电子",
                "银行",
                "家用电器",
                "综合",
                "煤炭",
                "房地产",
                "钢铁",
                "国防军工",
                "机械设备",
                "通信",
                "石油石化",
                "商贸零售",
                "食品饮料",
                "建筑材料",
                "建筑装饰",
                "汽车",
                "计算机",
                "有色金属",
                "非银金融",
                "医药生物",
                "美容护理",
                "基础化工",
                "电力设备",
                "公用事业",
                "纺织服饰",
                "社会服务",
                "环保",
                "传媒",
            ],
        }
    ).sort_values(["indus_we_cant_same"])
    if col_name:
        names = names.rename(columns={"indus_we_cant_same": col_name})
        df = pd.merge(df, names, on=[col_name])
    else:
        df = df.reset_index()
        df = df.rename(columns={list(df.columns)[0]: "indus_we_cant_same"})
        df = (
            pd.merge(df, names, on=["indus_we_cant_same"])
            .set_index("行业名称")
            .drop(columns=["indus_we_cant_same"])
        )
    return df


def rqdatac_show_used() -> float:
    """查询流量使用情况

    Returns
    -------
    `float`
        当日已经使用的流量MB数
    """
    user2 = round(rqdatac.user.get_quota()["bytes_used"] / 1024 / 1024, 2)
    print(f"今日已使用rqsdk流量{user2}MB")
    return user2


def add_suffix(code: str) -> str:
    """给没有后缀的股票代码加上wind后缀

    Parameters
    ----------
    code : str
        没有后缀的股票代码

    Returns
    -------
    `str`
        加完wind后缀的股票代码
    """
    if code.startswith("0") or code.startswith("3"):
        code = code + ".SZ"
    elif code.startswith("6"):
        code = code + ".SH"
    elif code.startswith("8"):
        code = code + ".BJ"
    else:
        code = code + ".UN"
    return code


def 生成每日分类表(
    df: pd.DataFrame, code: str, entry: str, exit: str, kind: str
) -> pd.DataFrame:
    """
    ```
    df是要包含任意多列的表格，为dataframe格式，主要内容为，每一行是
    一只股票或一只基金的代码、分类、进入该分类的时间、移除该分类的时间，
    除此之外，还可以包含很多其他内容
    code是股票代码列的列名，为字符串格式；
    entry是股票进入该分类的日期的列名，为字符串格式
    exit是股票退出该分类的日期的列名，为字符串格式
    kind是分类列的列名，为字符串格式
    ```
    """
    df = df[[code, entry, exit, kind]]
    df = df.fillna(int(datetime.datetime.now().strftime("%Y%m%d")))
    try:
        if type(df[entry].iloc[0]) == str:
            df[entry] = df[entry].astype(str)
            df[exit] = df[exit].astype(str)
        else:
            df[entry] = df[entry].astype(int).astype(str)
            df[exit] = df[exit].astype(int).astype(str)
    except Exception:
        print("您的进入日期和推出日期，既不是字符串，又不是数字格式，好好检查一下吧")
    df = df.set_index([code, kind])
    df = df.stack().to_frame(name="date")

    def fill_middle(df1):
        min_time = df1.date.min()
        max_time = df1.date.max()
        df2 = pd.DataFrame({"date": pd.date_range(min_time, max_time)})
        return df2

    ff = df.reset_index().groupby([code, kind]).apply(fill_middle)
    ff = ff.reset_index()
    ff = ff[[code, kind, "date"]]
    ff = ff[ff.date >= pd.Timestamp("2004-01-01")]
    return ff


def set_index_first(df: pd.DataFrame) -> pd.DataFrame:
    """将dataframe的第一列，无论其是什么名字，都设置为index

    Parameters
    ----------
    df : pd.DataFrame
        要修改的dataframe
    Returns
    -------
    pd.DataFrame
        修改后的dataframe
    """
    df = df.set_index(list(df.columns)[0])
    return df


def change_index_name(df: pd.DataFrame, name: str = "date") -> pd.DataFrame:
    """修改dataframe的index的名称，便于写入feather时统一命名

    Parameters
    ----------
    df : pd.DataFrame
        要修改的dataframe
    name : str, optional
        想要修改的名字, by default 'date'

    Returns
    -------
    pd.DataFrame
        修改后的dataframe
    """
    df = df.reset_index()
    df.columns = [name] + list(df.columns)[1:]
    df = set_index_first(df)
    return df


def merge_many(dfs: list[pd.DataFrame], names: list = None) -> pd.DataFrame:
    """将多个宽dataframe依据columns和index，拼接在一起，拼成一个长dataframe

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        将所有要拼接的宽表放在一个列表里
    names : list, optional
        拼接后，每一列宽表对应的名字, by default None

    Returns
    -------
    pd.DataFrame
        拼接后的dataframe
    """
    num = len(dfs)
    if names is None:
        names = [f"fac{i+1}" for i in range(num)]
    dfs = [i.stack().reset_index() for i in dfs]
    dfs = [i.rename(columns={list(i.columns)[-1]: j}) for i, j in zip(dfs, names)]
    dfs = [i.rename(columns={list(i.columns)[-2]: "code"}) for i in dfs]
    df = reduce(lambda x, y: pd.merge(x, y, on=["date", "code"]), dfs)
    return df


def corr_two_daily(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    history: str = None,
    rolling_window: int = 20,
    n_jobs: int = 6,
) -> pd.DataFrame:
    """求两个因子，在相同股票上，时序上滚动窗口下的相关系数

    Parameters
    ----------
    df1 : pd.DataFrame
        第一个因子，index为时间，columns为股票代码
    df2 : pd.DataFrame
        第二个因子，index为时间，columns为股票代码
    history : str, optional
        从某处读取计算好的历史文件
    rolling_window : int, optional
        滚动窗口, by default 20
    n_jobs : int, optional
        并行数量, by default 6

    Returns
    -------
    pd.DataFrame
        相关系数后的结果，index为时间，columns为股票代码
    """

    def corr_in(a, b, c):
        return c.iloc[-1], np.corrcoef(a, b)[0, 1]

    return func_two_daily(
        df1=df1,
        df2=df2,
        func=corr_in,
        history=history,
        rolling_window=rolling_window,
        n_jobs=n_jobs,
    )


def cov_two_daily(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    history: str = None,
    rolling_window: int = 20,
    n_jobs: int = 6,
) -> pd.DataFrame:
    """求两个因子，在相同股票上，时序上滚动窗口下的协方差

    Parameters
    ----------
    df1 : pd.DataFrame
        第一个因子，index为时间，columns为股票代码
    df2 : pd.DataFrame
        第二个因子，index为时间，columns为股票代码
    history : str, optional
        从某处读取计算好的历史文件
    rolling_window : int, optional
        滚动窗口, by default 20
    n_jobs : int, optional
        并行数量, by default 6

    Returns
    -------
    pd.DataFrame
        求协方差后的结果，index为时间，columns为股票代码
    """

    def cov_in(a, b, c):
        return c.iloc[-1], np.cov(a, b)[0, 1]

    return func_two_daily(
        df1=df1,
        df2=df2,
        func=cov_in,
        history=history,
        rolling_window=rolling_window,
        n_jobs=n_jobs,
    )


def func_two_daily(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    func: Callable,
    history: str = None,
    rolling_window: int = 20,
    n_jobs: int = 6,
) -> pd.DataFrame:
    """求两个因子，在相同股票上，时序上滚动窗口下的相关系数

    Parameters
    ----------
    df1 : pd.DataFrame
        第一个因子，index为时间，columns为股票代码
    df2 : pd.DataFrame
        第二个因子，index为时间，columns为股票代码
    func : Callable
        要对两列数进行操作的函数
    history : str, optional
        从某处读取计算好的历史文件
    rolling_window : int, optional
        滚动窗口, by default 20
    n_jobs : int, optional
        并行数量, by default 6

    Returns
    -------
    pd.DataFrame
        计算后的结果，index为时间，columns为股票代码
    """

    the_func = partial(func)

    def func_rolling(df):
        df = df.sort_values(["date"])
        if df.shape[0] > rolling_window:
            df = npext.rolling_apply(
                the_func, rolling_window, df.fac1, df.fac2, df.date, n_jobs=n_jobs
            )
            return df

    homeplace = HomePlace()
    if history is not None:
        if os.path.exists(homeplace.update_data_file + history):
            old = pd.read_feather(homeplace.update_data_file + history)
            old = old.set_index(list(old.columns)[0])
            new_end = min(df1.index.max(), df2.index.max())
            if new_end > old.index.max():
                old_end = datetime.datetime.strftime(old.index.max(), "%Y%m%d")
                logger.info(f"上次更新到了{old_end}")
                df1a = df1[df1.index <= old.index.max()].tail(rolling_window - 1)
                df1b = df1[df1.index > old.index.max()]
                df1 = pd.concat([df1a, df1b])
                df2a = df2[df2.index <= old.index.max()].tail(rolling_window - 1)
                df2b = df2[df2.index > old.index.max()]
                df2 = pd.concat([df2a, df2b])
                twins = merge_many([df1, df2])

                if is_notebook():
                    tqdm.tqdm_notebook().pandas()
                else:
                    tqdm.tqdm.pandas()
                corrs = twins.groupby(["code"]).progress_apply(func_rolling)
                cor = []
                for i in range(len(corrs)):
                    df = (
                        pd.DataFrame(corrs.iloc[i]).dropna().assign(code=corrs.index[i])
                    )
                    cor.append(df)
                cors = pd.concat(cor)
                cors.columns = ["date", "corr", "code"]
                cors = cors.pivot(index="date", columns="code", values="corr")
                if history is not None:
                    if os.path.exists(homeplace.update_data_file + history):
                        cors = pd.concat([old, cors])
                    cors = drop_duplicates_index(cors)
                    cors.reset_index().to_feather(homeplace.update_data_file + history)
                    new_end = datetime.datetime.strftime(cors.index.max(), "%Y%m%d")
                    logger.info(f"已经更新至{new_end}")
                return cors
            else:
                logger.info(f"已经是最新的了")
                return old
        else:
            logger.info("第一次计算，请耐心等待，计算完成后将存储")
            twins = merge_many([df1, df2])
            if is_notebook():
                tqdm.tqdm_notebook().pandas()
            else:
                tqdm.tqdm.pandas()
            corrs = twins.groupby(["code"]).progress_apply(func_rolling)
            cor = []
            for i in range(len(corrs)):
                df = pd.DataFrame(corrs.iloc[i]).dropna().assign(code=corrs.index[i])
                cor.append(df)
            cors = pd.concat(cor)
            cors.columns = ["date", "corr", "code"]
            cors = cors.pivot(index="date", columns="code", values="corr")
            if history is not None:
                if os.path.exists(homeplace.update_data_file + history):
                    cors = pd.concat([old, cors])
                cors = drop_duplicates_index(cors)
                cors.reset_index().to_feather(homeplace.update_data_file + history)
                new_end = datetime.datetime.strftime(cors.index.max(), "%Y%m%d")
                logger.info(f"已经更新至{new_end}")
            return cors


def drop_duplicates_index(new: pd.DataFrame) -> pd.DataFrame:
    """对dataframe依照其index进行去重，并保留最上面的行

    Parameters
    ----------
    new : pd.DataFrame
        要去重的dataframe

    Returns
    -------
    pd.DataFrame
        去重后的dataframe
    """
    new = new.reset_index()
    new = new.rename(columns={list(new.columns)[0]: "date"})
    new = new.drop_duplicates(subset=["date"], keep="first")
    new = new.set_index("date")
    return new


def select_max(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """两个columns与index完全相同的df，每个值都挑出较大值

    Parameters
    ----------
    df1 : pd.DataFrame
        第一个df
    df2 : pd.DataFrame
        第二个df

    Returns
    -------
    `pd.DataFrame`
        两个df每个value中的较大者
    """
    return (df1 + df2 + np.abs(df1 - df2)) / 2


def select_min(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """两个columns与index完全相同的df，每个值都挑出较小值

    Parameters
    ----------
    df1 : pd.DataFrame
        第一个df
    df2 : pd.DataFrame
        第二个df

    Returns
    -------
    `pd.DataFrame`
        两个df每个value中的较小者
    """
    return (df1 + df2 - np.abs(df1 - df2)) / 2


def debj(df: pd.DataFrame) -> pd.DataFrame:
    """去除因子中的北交所数据

    Parameters
    ----------
    df : pd.DataFrame
        包含北交所的因子dataframe，index是时间，columns是股票代码

    Returns
    -------
    pd.DataFrame
        去除北交所股票的因子dataframe
    """
    df = df[[i for i in list(df.columns) if i[0] in ["0", "3", "6"]]]
    return df


def standardlize(df: pd.DataFrame, all_pos: bool = 0) -> pd.DataFrame:
    """对因子dataframe做横截面z-score标准化

    Parameters
    ----------
    df : pd.DataFrame
        要做中性化的因子值，index是时间，columns是股票代码
    all_pos : bool, optional
        是否要将值都变成正数，通过减去截面的最小值实现, by default 0

    Returns
    -------
    pd.DataFrame
        标准化之后的因子
    """
    df = ((df.T - df.T.mean()) / df.T.std()).T
    if all_pos:
        df = (df.T - df.T.min()).T
    return df


def count_value(df: pd.DataFrame, with_zero: bool = 0) -> int:
    """计算dataframe中总共有多少（非0）非空的值

    Parameters
    ----------
    df : pd.DataFrame
        要检测的dataframe
    with_zero : bool, optional
        统计数量时，是否也把值为0的数据统计进去, by default 0

    Returns
    -------
    int
        （非0）非空的数据的个数
    """
    y = np.sign(np.abs(df))
    if with_zero:
        y = np.sign(y + 1)
    return y.sum().sum()


def detect_nan(df: pd.DataFrame) -> bool:
    """检查一个pd.DataFrame中是否存在空值

    Parameters
    ----------
    df : pd.DataFrame
        待检查的pd.DataFrame

    Returns
    -------
    `bool`
        检查结果，有空值为True，否则为False
    """
    x = df.isna() + 0
    if x.sum().sum():
        print("存在空值")
        return True
    else:
        print("不存在空值")
        return False


def get_abs(df: pd.DataFrame, median: bool = 0, square: bool = 0) -> pd.DataFrame:
    """均值距离化：计算因子与截面均值的距离

    Parameters
    ----------
    df : pd.DataFrame
        未均值距离化的因子，index为时间，columns为股票代码
    median : bool, optional
        为1则计算到中位数的距离, by default 0
    square : bool, optional
        为1则计算距离的平方, by default 0

    Returns
    -------
    `pd.DataFrame`
        均值距离化之后的因子值
    """
    if not square:
        if median:
            return np.abs((df.T - df.T.median()).T)
        else:
            return np.abs((df.T - df.T.mean()).T)
    else:
        if median:
            return ((df.T - df.T.median()).T) ** 2
        else:
            return ((df.T - df.T.mean()).T) ** 2


def get_normal(df: pd.DataFrame) -> pd.DataFrame:
    """将因子横截面正态化

    Parameters
    ----------
    df : pd.DataFrame
        原始因子，index是时间，columns是股票代码

    Returns
    -------
    `pd.DataFrame`
        每个横截面都呈现正态分布的因子
    """
    df = df.replace(0, np.nan)
    df = df.T.apply(lambda x: ss.boxcox(x)[0]).T
    return df


def coin_reverse(
    ret20: pd.DataFrame, vol20: pd.DataFrame, mean: bool = 1, positive_negtive: bool = 0
) -> pd.DataFrame:
    """球队硬币法：根据vol20的大小，翻转一半ret20，把vol20较大的部分，给ret20添加负号

    Parameters
    ----------
    ret20 : pd.DataFrame
        要被翻转的因子，index是时间，columns是股票代码
    vol20 : pd.DataFrame
        翻转的依据，index是时间，columns是股票代码
    mean : bool, optional
        为1则以是否大于截面均值为标准翻转，否则以是否大于截面中位数为标准, by default 1
    positive_negtive : bool, optional
        是否截面上正负值的两部分，各翻转一半, by default 0

    Returns
    -------
    `pd.DataFrame`
        翻转后的因子值
    """
    if positive_negtive:
        if not mean:
            down20 = np.sign(ret20)
            down20 = down20.replace(1, np.nan)
            down20 = down20.replace(-1, 1)

            vol20_down = down20 * vol20
            vol20_down = (vol20_down.T - vol20_down.T.median()).T
            vol20_down = np.sign(vol20_down)
            ret20_down = ret20[ret20 < 0]
            ret20_down = vol20_down * ret20_down

            up20 = np.sign(ret20)
            up20 = up20.replace(-1, np.nan)

            vol20_up = up20 * vol20
            vol20_up = (vol20_up.T - vol20_up.T.median()).T
            vol20_up = np.sign(vol20_up)
            ret20_up = ret20[ret20 > 0]
            ret20_up = vol20_up * ret20_up

            ret20_up = ret20_up.replace(np.nan, 0)
            ret20_down = ret20_down.replace(np.nan, 0)
            new_ret20 = ret20_up + ret20_down
            new_ret20_tr = new_ret20.replace(0, np.nan)
            return new_ret20_tr
        else:
            down20 = np.sign(ret20)
            down20 = down20.replace(1, np.nan)
            down20 = down20.replace(-1, 1)

            vol20_down = down20 * vol20
            vol20_down = (vol20_down.T - vol20_down.T.mean()).T
            vol20_down = np.sign(vol20_down)
            ret20_down = ret20[ret20 < 0]
            ret20_down = vol20_down * ret20_down

            up20 = np.sign(ret20)
            up20 = up20.replace(-1, np.nan)

            vol20_up = up20 * vol20
            vol20_up = (vol20_up.T - vol20_up.T.mean()).T
            vol20_up = np.sign(vol20_up)
            ret20_up = ret20[ret20 > 0]
            ret20_up = vol20_up * ret20_up

            ret20_up = ret20_up.replace(np.nan, 0)
            ret20_down = ret20_down.replace(np.nan, 0)
            new_ret20 = ret20_up + ret20_down
            new_ret20_tr = new_ret20.replace(0, np.nan)
            return new_ret20_tr
    else:
        if not mean:
            vol20_dummy = np.sign((vol20.T - vol20.T.median()).T)
            ret20 = ret20 * vol20_dummy
            return ret20
        else:
            vol20_dummy = np.sign((vol20.T - vol20.T.mean()).T)
            ret20 = ret20 * vol20_dummy
            return ret20


def multidfs_to_one(*args: list) -> pd.DataFrame:
    """很多个df，各有一部分，其余位置都是空，
    想把各自df有值的部分保留，都没有值的部分继续设为空

    Returns
    -------
    `pd.DataFrame`
        合并后的df
    """
    dfs = [i.fillna(0) for i in args]
    background = np.sign(np.abs(np.sign(sum(dfs))) + 1).replace(1, 0)
    dfs = [(i + background).fillna(0) for i in dfs]
    df_nans = [i.isna() for i in dfs]
    nan = reduce(lambda x, y: x * y, df_nans)
    nan = nan.replace(1, np.nan)
    nan = nan.replace(0, 1)
    df_final = sum(dfs) * nan
    return df_final


def to_percent(x: float) -> Union[float, str]:
    """把小数转化为2位小数的百分数

    Parameters
    ----------
    x : float
        要转换的小数

    Returns
    -------
    Union[float,str]
        空值则依然为空，否则返回带%的字符串
    """
    if np.isnan(x):
        return x
    else:
        x = str(round(x * 100, 2)) + "%"
        return x


def calc_exp_list(window: int, half_life: int) -> np.ndarray:
    """生成半衰序列

    Parameters
    ----------
    window : int
        窗口期
    half_life : int
        半衰期

    Returns
    -------
    `np.ndarray`
        半衰序列
    """
    exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
    return exp_wt[::-1] / np.sum(exp_wt)


def calcWeightedStd(series: pd.Series, weights: Union[pd.Series, np.ndarray]) -> float:
    """计算半衰加权标准差

    Parameters
    ----------
    series : pd.Series
        目标序列
    weights : Union[pd.Series,np.ndarray]
        权重序列

    Returns
    -------
    `float`
        半衰加权标准差
    """
    weights /= np.sum(weights)
    return np.sqrt(np.sum((series - np.mean(series)) ** 2 * weights))


def get_list_std(delta_sts: list[pd.DataFrame]) -> pd.DataFrame:
    """同一天多个因子，计算这些因子在当天的标准差

    Parameters
    ----------
    delta_sts : list[pd.DataFrame]
        多个因子构成的list，每个因子index为时间，columns为股票代码

    Returns
    -------
    `pd.DataFrame`
        每天每只股票多个因子的标准差
    """
    delta_sts_mean = sum(delta_sts) / len(delta_sts)
    delta_sts_std = [(i - delta_sts_mean) ** 2 for i in delta_sts]
    delta_sts_std = sum(delta_sts_std)
    delta_sts_std = delta_sts_std**0.5 / len(delta_sts)
    return delta_sts_std


def to_group(df: pd.DataFrame, group: int = 10) -> pd.DataFrame:
    """把一个index为时间，code为时间的df，每个截面上的值，按照排序分为group组，将值改为组号，从0开始

    Parameters
    ----------
    df : pd.DataFrame
        要改为组号的df
    group : int, optional
        分为多少组, by default 10

    Returns
    -------
    pd.DataFrame
        组号组成的dataframe
    """
    df = df.T.apply(lambda x: pd.qcut(x, group, labels=False, duplicates="drop")).T
    return df


def same_columns(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """保留多个dataframe共同columns的部分

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        多个dataframe

    Returns
    -------
    list[pd.DataFrame]
        保留共同部分后的结果
    """
    dfs = [i.T for i in dfs]
    res = []
    for i, df in enumerate(dfs):

        others = dfs[:i] + dfs[i + 1 :]

        for other in others:
            df = df[df.index.isin(other.index)]
        res.append(df.T)
    return res


def same_index(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """保留多个dataframe共同index的部分

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        多个dataframe

    Returns
    -------
    list[pd.DataFrame]
        保留共同部分后的结果
    """
    res = []
    for i, df in enumerate(dfs):

        others = dfs[:i] + dfs[i + 1 :]

        for other in others:
            df = df[df.index.isin(other.index)]
        res.append(df)
    return res


def feather_to_parquet(folder: str):
    """将某个路径下的所有feather文件都转化为parquet文件

    Parameters
    ----------
    folder : str
        要转化的文件夹路径
    """
    files = os.listdir(folder)
    files = [folder + i for i in files]
    if is_notebook():
        for file in tqdm.tqdm_notebook(files):
            try:
                df = pd.read_feather(file)
                if (
                    ("date" in list(df.columns)) and ("code" not in list(df.columns))
                ) or ("index" in list(df.columns)):
                    df = df.set_index(list(df.columns)[0])
                df.to_parquet(file.split(".")[0]+'.parquet')
            except Exception:
                logger.warning(f"{file}不是parquet文件")
