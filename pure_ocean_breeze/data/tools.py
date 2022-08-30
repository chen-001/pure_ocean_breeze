"""
针对一些不常见的文件格式，读取数据文件的一些工具函数，以及其他数据工具
"""

__updated__ = "2022-08-30 18:03:58"

import h5py
import pandas as pd
import tqdm
import datetime
import scipy.io as scio
import numpy as np

try:
    import rqdatac

    rqdatac.init()
except Exception:
    print("暂时未连接米筐")


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
