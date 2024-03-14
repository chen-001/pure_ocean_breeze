__updated__ = "2022-11-05 00:16:10"

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt

plt.style.use(["science", "no-latex", "notebook"])
plt.rcParams["axes.unicode_minus"] = False
from pure_ocean_breeze.legacy_version.v3p4.data.read_data import read_index_three, read_daily


def comment_on_rets_and_nets(
    rets: pd.Series, nets: pd.Series, name: str = "绩效"
) -> pd.DataFrame:
    """输入月频的收益率序列和净值序列，输出年化收益、年化波动、信息比率、月度胜率和最大回撤率
    输入2个pd.Series，时间是索引

    Parameters
    ----------
    rets : pd.Series
        收益率序列，index为时间
    nets : pd.Series
        净值序列，index为时间
    name : str, optional
        绩效指标列名字, by default '绩效'

    Returns
    -------
    `pd.DataFrame`
        包含年化收益、年化波动、信息比率、月度胜率和最大回撤率的评价指标
    """
    duration_nets = (nets.index[-1] - nets.index[0]).days
    year_nets = duration_nets / 365
    ret_yearly = (nets.iloc[-1] / nets.iloc[0]) ** (1 / year_nets) - 1
    max_draw = ((nets.cummax() - nets) / nets.cummax()).max()
    vol = np.std(rets) * (12**0.5)
    info_rate = ret_yearly / vol
    win_rate = len(rets[rets > 0]) / len(rets)
    comments = pd.DataFrame(
        {
            "年化收益率": ret_yearly,
            "年化波动率": vol,
            "信息比率": info_rate,
            "月度胜率": win_rate,
            "最大回撤率": max_draw,
        },
        index=[name],
    ).T
    return comments


def comments_on_twins(nets: pd.Series, rets: pd.Series) -> pd.Series:
    """输入月频的收益率序列和净值序列，给出评价
    评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率
    输入2个pd.Series，时间是索引

    Parameters
    ----------
    nets : pd.Series
        净值序列，index为时间
    rets : pd.Series
        收益率序列，index为时间

    Returns
    -------
    `pd.Series`
        包含年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率的评价指标
    """
    series = nets.copy()
    series1 = rets.copy()
    ret = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
    duration = (series.index[-1] - series.index[0]).days
    year = duration / 365
    ret_yearly = (series.iloc[-1] / series.iloc[0]) ** (1 / year) - 1
    max_draw = -(series / series.expanding(1).max() - 1).min()
    vol = np.std(series1) * (12**0.5)
    sharpe = ret_yearly / vol
    wins = series1[series1 > 0]
    win_rate = len(wins) / len(series1)
    return pd.Series(
        [ret, ret_yearly, vol, sharpe, win_rate, max_draw],
        index=["总收益率", "年化收益率", "年化波动率", "信息比率", "胜率", "最大回撤率"],
    )


def comments_on_twins_periods(
    nets: pd.Series, rets: pd.Series, periods: int
) -> pd.Series:
    """输入其他频率的收益率序列和净值序列，给出评价
    评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率
    输入2个pd.Series，时间是索引

    Parameters
    ----------
    nets : pd.Series
        净值序列，index为时间
    rets : pd.Series
        收益率序列，index为时间
    periods : int
        收益率序列的频率，如5天一次，则为5

    Returns
    -------
    `pd.Series`
        包含年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率的评价指标
    """
    series = nets.copy()
    series1 = rets.copy()
    ret = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
    duration = (series.index[-1] - series.index[0]).days
    year = duration / 365
    ret_yearly = (series.iloc[-1] / series.iloc[0]) ** (1 / year) - 1
    max_draw = -(series / series.expanding(1).max() - 1).min()
    vol = np.std(series1) * (252**0.5) * (periods**0.5)
    sharpe = ret_yearly / vol
    wins = series1[series1 > 0]
    win_rate = len(wins) / len(series1)
    return pd.Series(
        [ret, ret_yearly, vol, sharpe, win_rate, max_draw],
        index=["总收益率", "年化收益率", "年化波动率", "信息比率", "胜率", "最大回撤率"],
    )


def make_relative_comments(
    ret_fac: pd.Series,
    hs300: bool = 0,
    zz500: bool = 0,
    zz1000: bool = 0,
    gz2000: bool = 0,
    day: int = None,
    show_nets: bool = 0,
) -> pd.Series:
    """对于一个给定的收益率序列，计算其相对于某个指数的超额表现

    Parameters
    ----------
    ret_fac : pd.Series
        给定的收益率序列，index为时间
    hs300 : bool, optional
        为1则相对沪深300指数行情, by default 0
    zz500 : bool, optional
        为1则相对中证500指数行情, by default 0
    zz1000 : bool, optional
        为1则相对中证1000指数行情, by default 0
    gz2000 : bool, optional
        为1则相对国证2000指数行情, by default 0
    day : int, optional
        起始日期，形如20130101, by default None
    show_nets : bool, optional
        返回值中包括超额净值数据, by default 0

    Returns
    -------
    `pd.Series`
        评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率

    Raises
    ------
    `IOError`
        如果没指定任何一个指数，将报错
    """
    if hs300:
        net_index = read_index_three(day=day)[0]
    elif zz500:
        net_index = read_index_three(day=day)[1]
    elif zz1000:
        net_index = read_index_three(day=day)[2]
    elif gz2000:
        net_index = read_index_three(day=day)[3]
    else:
        raise IOError("你总得指定一个股票池吧？")
    ret_index = net_index.pct_change()
    ret = ret_fac - ret_index
    ret = ret.dropna()
    net = (1 + ret).cumprod()
    ntop = pd.Series(1, index=[net.index.min() - pd.DateOffset(months=1)])
    rtop = pd.Series(0, index=[net.index.min() - pd.DateOffset(months=1)])
    net = pd.concat([ntop, net]).resample("M").last()
    ret = pd.concat([rtop, ret]).resample("M").last()
    com = comments_on_twins(net, ret)
    if show_nets:
        return com, net
    else:
        return com


def make_relative_comments_plot(
    ret_fac: pd.Series,
    hs300: bool = 0,
    zz500: bool = 0,
    zz1000: bool = 0,
    gz2000: bool = 0,
    day: int = None,
) -> pd.Series:
    """对于一个给定的收益率序列，计算其相对于某个指数的超额表现，然后绘图，并返回超额净值序列

    Parameters
    ----------
    ret_fac : pd.Series
        给定的收益率序列，index为时间
    hs300 : bool, optional
        为1则相对沪深300指数行情, by default 0
    zz500 : bool, optional
        为1则相对中证500指数行情, by default 0
    zz1000 : bool, optional
        为1则相对中证1000指数行情, by default 0
    gz2000 : bool, optional
        为1则相对国证2000指数行情, by default 0
    day : int, optional
        起始日期，形如20130101, by default None

    Returns
    -------
    `pd.Series`
        超额净值序列

    Raises
    ------
    `IOError`
        如果没指定任何一个指数，将报错
    """
    if hs300:
        net_index = read_index_three(day=day)[0]
    elif zz500:
        net_index = read_index_three(day=day)[1]
    elif zz1000:
        net_index = read_index_three(day=day)[2]
    elif gz2000:
        net_index = read_index_three(day=day)[3]
    else:
        raise IOError("你总得指定一个股票池吧？")
    ret_index = net_index.pct_change()
    ret = ret_fac - ret_index
    ret = ret.dropna()
    net = (1 + ret).cumprod()
    ntop = pd.Series(1, index=[net.index.min() - pd.DateOffset(months=1)])
    rtop = pd.Series(0, index=[net.index.min() - pd.DateOffset(months=1)])
    net = pd.concat([ntop, net]).resample("M").last()
    ret = pd.concat([rtop, ret]).resample("M").last()
    com = comments_on_twins(net, ret)
    net.plot(rot=60)
    plt.show()
    return net


def other_periods_comments_nets(
    fac: pd.DataFrame,
    way: str,
    period: int,
    comments_writer: pd.ExcelWriter = None,
    nets_writer: pd.ExcelWriter = None,
    sheetname: str = None,
    group_num: int = 10,
) -> tuple[pd.Series]:
    """小型回测框架，不同频率下的评价指标，请输入行业市值中性化后的因子值

    Parameters
    ----------
    fac : pd.DataFrame
        行业市值中性化之后的因子值，index为时间，columns为股票代码
    way : str
        因子的方向，可选值 pos 或 neg
    period : int
        频率，如5天则为5, by default None
    comments_writer : pd.ExcelWriter, optional
        写入绩效的xlsx, by default None
    nets_writer : pd.ExcelWriter, optional
        写入净值的xlsx, by default None
    sheetname : str, optional
        工作表名称, by default None
    group_num : int, optional
        回测时分组数量, by default 10

    Returns
    -------
    `tuple[pd.Series]`
        绩效和净值
    """
    import alphalens as al

    closes = read_daily(open=1).shift(-1)
    fac1 = fac.stack()
    df = al.utils.get_clean_factor_and_forward_returns(
        fac1,
        closes[closes.index.isin(fac.index)],
        quantiles=group_num,
        periods=(period,),
    )
    df = df.reset_index()
    ics = df.groupby(["date"])[[f"{period}D", "factor"]].apply(
        lambda x: x.corr(method="spearman").iloc[0, 1]
    )
    ic = ics.mean()
    ir = ics.std()
    icir = ic / ir * (252**0.5) / (period**0.5)
    df = df.groupby(["date", "factor_quantile"])[f"{period}D"].mean() / period
    df = df.unstack()
    df.columns = [f"分组{i}" for i in list(df.columns)]
    if way == "pos":
        df = df.assign(多空对冲=df[f"分组{group_num}"] - df.分组1)
    elif way == "neg":
        df = df.assign(多空对冲=df.分组1 - df[f"分组{group_num}"])
    nets = (df + 1).cumprod()
    nets = nets.apply(lambda x: x / x.iloc[0])
    nets.plot(rot=60)
    plt.show()
    comments = comments_on_twins_periods(nets.多空对冲, df.多空对冲, period)
    comments = pd.concat(
        [pd.Series([ic, icir], index=["Rank IC", "Rank ICIR"]), comments]
    )
    print(comments)
    if sheetname is None:
        ...
    else:
        if comments_writer is None:
            ...
        else:
            comments.to_excel(comments_writer, sheetname)
        if nets_writer is None:
            ...
        else:
            nets.to_excel(nets_writer, sheetname)
    return comments, nets
