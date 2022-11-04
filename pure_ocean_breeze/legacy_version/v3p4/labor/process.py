__updated__ = "2022-11-05 00:16:56"

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import knockknock as kk
import os
import tqdm
import scipy.stats as ss
import statsmodels.formula.api as smf
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt

plt.style.use(["science", "no-latex", "notebook"])
plt.rcParams["axes.unicode_minus"] = False

from functools import reduce, lru_cache, partial
from dateutil.relativedelta import relativedelta
from loguru import logger
import datetime
from collections.abc import Iterable
import plotly.express as pe
import plotly.io as pio
from plotly.tools import FigureFactory as FF
import plotly.graph_objects as go
import plotly.tools as plyoo
import pyfinance.ols as po
from texttable import Texttable
from xpinyin import Pinyin
import cufflinks as cf

cf.set_config_file(offline=True)
from typing import Callable, Union
from pure_ocean_breeze.legacy_version.v3p4.data.read_data import (
    read_daily,
    read_market,
    get_industry_dummies,
    read_swindustry_prices,
    read_zxindustry_prices,
    database_read_final_factors,
)
from pure_ocean_breeze.legacy_version.v3p4.state.homeplace import HomePlace

homeplace = HomePlace()
from pure_ocean_breeze.legacy_version.v3p4.state.states import STATES, is_notebook
from pure_ocean_breeze.legacy_version.v3p4.data.database import *
from pure_ocean_breeze.legacy_version.v3p4.data.dicts import INDUS_DICT
from pure_ocean_breeze.legacy_version.v3p4.data.tools import (
    indus_name,
    drop_duplicates_index,
    to_percent,
    to_group,
)
from pure_ocean_breeze.legacy_version.v3p4.labor.comment import (
    comments_on_twins,
    make_relative_comments,
    make_relative_comments_plot,
)


def daily_factor_on300500(
    fac: pd.DataFrame,
    hs300: bool = 0,
    zz500: bool = 0,
    zz800: bool = 0,
    zz1000: bool = 0,
    gz2000: bool = 0,
    other: bool = 0,
) -> pd.DataFrame:
    """输入日频或月频因子值，将其限定在某指数成分股的股票池内，
    目前仅支持沪深300、中证500、中证800、中证1000、国证2000成分股，和除沪深300、中证500、中证1000以外的股票的成分股

    Parameters
    ----------
    fac : pd.DataFrame
        未限定股票池的因子值，index为时间，columns为股票代码
    hs300 : bool, optional
        限定股票池为沪深300, by default 0
    zz500 : bool, optional
        限定股票池为中证500, by default 0
    zz800 : bool, optional
        限定股票池为中证800, by default 0
    zz1000 : bool, optional
        限定股票池为中证1000, by default 0
    gz2000 : bool, optional
        限定股票池为国证2000, by default 0
    other : bool, optional
        限定股票池为除沪深300、中证500、中证1000以外的股票的成分股, by default 0

    Returns
    -------
    `pd.DataFrame`
        仅包含成分股后的因子值，非成分股的因子值为空

    Raises
    ------
    `ValueError`
        如果未指定任何一种指数的成分股，将报错
    """
    last = fac.resample("M").last()
    homeplace = HomePlace()
    if fac.shape[0] / last.shape[0] > 2:
        if hs300:
            df = pd.read_feather(
                homeplace.daily_data_file + "沪深300日成分股.feather"
            ).replace(0, np.nan)
            df = df.set_index(list(df.columns)[0])
            df = df * fac
            df = df.dropna(how="all")
        elif zz500:
            df = pd.read_feather(
                homeplace.daily_data_file + "中证500日成分股.feather"
            ).replace(0, np.nan)
            df = df.set_index(list(df.columns)[0])
            df = df * fac
            df = df.dropna(how="all")
        elif zz800:
            df1 = pd.read_feather(homeplace.daily_data_file + "沪深300日成分股.feather")
            df1 = df1.set_index(list(df1.columns)[0])
            df2 = pd.read_feather(homeplace.daily_data_file + "中证500日成分股.feather")
            df2 = df2.set_index(list(df2.columns)[0])
            df = df1 + df2
            df = df.replace(0, np.nan)
            df = df * fac
            df = df.dropna(how="all")
        elif zz1000:
            df = pd.read_feather(
                homeplace.daily_data_file + "中证1000日成分股.feather"
            ).replace(0, np.nan)
            df = df.set_index(list(df.columns)[0])
            df = df * fac
            df = df.dropna(how="all")
        elif gz2000:
            df = pd.read_feather(
                homeplace.daily_data_file + "国证2000日成分股.feather"
            ).replace(0, np.nan)
            df = df.set_index(list(df.columns)[0])
            df = df * fac
            df = df.dropna(how="all")
        elif other:
            tr = read_daily(tr=1).fillna(0).replace(0, 1)
            tr = np.sign(tr)
            df1 = (
                tr * pd.read_feather(homeplace.daily_data_file + "沪深300日成分股.feather")
            ).fillna(0)
            df1 = df1.set_index(list(df1.columns)[0])
            df2 = (
                tr * pd.read_feather(homeplace.daily_data_file + "中证500日成分股.feather")
            ).fillna(0)
            df2 = df2.set_index(list(df2.columns)[0])
            df3 = (
                tr * pd.read_feather(homeplace.daily_data_file + "中证1000日成分股.feather")
            ).fillna(0)
            df3 = df3.set_index(list(df3.columns)[0])
            df = (1 - df1) * (1 - df2) * (1 - df3) * tr
            df = df.replace(0, np.nan) * fac
            df = df.dropna(how="all")
        else:
            raise ValueError("总得指定一下是哪个成分股吧🤒")
    else:
        if hs300:
            df = pd.read_feather(
                homeplace.daily_data_file + "沪深300日成分股.feather"
            ).replace(0, np.nan)
            df = df.set_index(list(df.columns)[0])
            df = df.resample("M").last()
            df = df * fac
            df = df.dropna(how="all")
        elif zz500:
            df = pd.read_feather(
                homeplace.daily_data_file + "中证500日成分股.feather"
            ).replace(0, np.nan)
            df = df.set_index(list(df.columns)[0])
            df = df.resample("M").last()
            df = df * fac
            df = df.dropna(how="all")
        elif zz800:
            df1 = pd.read_feather(homeplace.daily_data_file + "沪深300日成分股.feather")
            df1 = df1.set_index(list(df1.columns)[0])
            df1 = df1.resample("M").last()
            df2 = pd.read_feather(homeplace.daily_data_file + "中证500日成分股.feather")
            df2 = df2.set_index(list(df2.columns)[0])
            df2 = df2.resample("M").last()
            df = df1 + df2
            df = df.replace(0, np.nan)
            df = df * fac
            df = df.dropna(how="all")
        elif zz1000:
            df = pd.read_feather(
                homeplace.daily_data_file + "中证1000日成分股.feather"
            ).replace(0, np.nan)
            df = df.set_index(list(df.columns)[0])
            df = df.resample("M").last()
            df = df * fac
            df = df.dropna(how="all")
        elif gz2000:
            df = pd.read_feather(
                homeplace.daily_data_file + "国证2000日成分股.feather"
            ).replace(0, np.nan)
            df = df.set_index(list(df.columns)[0])
            df = df.resample("M").last()
            df = df * fac
            df = df.dropna(how="all")
        elif other:
            tr = read_daily(tr=1).fillna(0).replace(0, 1).resample("M").last()
            tr = np.sign(tr)
            df1 = (
                tr * pd.read_feather(homeplace.daily_data_file + "沪深300日成分股.feather")
            ).fillna(0)
            df1 = df1.set_index(list(df1.columns)[0])
            df1 = df1.resample("M").last()
            df2 = (
                tr * pd.read_feather(homeplace.daily_data_file + "中证500日成分股.feather")
            ).fillna(0)
            df2 = df2.set_index(list(df2.columns)[0])
            df2 = df2.resample("M").last()
            df3 = (
                tr * pd.read_feather(homeplace.daily_data_file + "中证1000日成分股.feather")
            ).fillna(0)
            df3 = df3.set_index(list(df3.columns)[0])
            df3 = df3.resample("M").last()
            df = (1 - df1) * (1 - df2) * (1 - df3)
            df = df.replace(0, np.nan) * fac
            df = df.dropna(how="all")
        else:
            raise ValueError("总得指定一下是哪个成分股吧🤒")
    return df


def daily_factor_on_industry(
    df: pd.DataFrame, swindustry: bool = 0, zxindustry: bool = 0
) -> dict:
    """将一个因子变为仅在某个申万一级行业上的股票

    Parameters
    ----------
    df : pd.DataFrame
        全市场的因子值，index是时间，columns是股票代码
    swindustry : bool, optional
        选择使用申万一级行业, by default 0
    zxindustry : bool, optional
        选择使用中信一级行业, by default 0

    Returns
    -------
    dict
        key为行业代码，value为对应的行业上的因子值
    """
    df1 = df.resample("M").last()
    if df1.shape[0] * 2 > df.shape[0]:
        daily = 0
        monthly = 1
    else:
        daily = 1
        monthly = 0
    start = int(datetime.datetime.strftime(df.index.min(), "%Y%m%d"))
    ress = get_industry_dummies(
        daily=daily,
        monthly=monthly,
        start=start,
        swindustry=swindustry,
        zxindustry=zxindustry,
    )
    ress = {k: v * df for k, v in ress.items()}
    return ress


def group_test_on_industry(
    df: pd.DataFrame,
    group_num: int = 10,
    net_values_writer: pd.ExcelWriter = None,
    swindustry: bool = 0,
    zxindustry: bool = 0,
) -> pd.DataFrame:
    """在申万一级行业上测试每个行业的分组回测

    Parameters
    ----------
    df : pd.DataFrame
        全市场的因子值，index是时间，columns是股票代码
    group_num : int, optional
        分组数量, by default 10
    net_values_writer : pd.ExcelWriter, optional
        用于存储各个行业分组及多空对冲净值序列的excel文件, by default None
    swindustry : bool, optional
        选择使用申万一级行业, by default 0
    zxindustry : bool, optional
        选择使用中信一级行业, by default 0

    Returns
    -------
    pd.DataFrame
        各个行业的绩效评价汇总
    """
    dfs = daily_factor_on_industry(df, swindustry=swindustry, zxindustry=zxindustry)

    ks = []
    vs = []
    if swindustry:
        for k, v in dfs.items():
            shen = pure_moonnight(
                v,
                groups_num=group_num,
                net_values_writer=net_values_writer,
                sheetname=INDUS_DICT[k],
                plt_plot=0,
            )
            ks.append(k)
            vs.append(shen.shen.total_comments.T)
        vs = pd.concat(vs)
        vs.index = [INDUS_DICT[i] for i in ks]
    else:
        for k, v in dfs.items():
            shen = pure_moonnight(
                v,
                groups_num=group_num,
                net_values_writer=net_values_writer,
                sheetname=k,
                plt_plot=0,
            )
            ks.append(k)
            vs.append(shen.shen.total_comments.T)
        vs = pd.concat(vs)
        vs.index = ks
    return vs


def rankic_test_on_industry(
    df: pd.DataFrame,
    excel_name: str = "行业rankic.xlsx",
    png_name: str = "行业rankic图.png",
    swindustry: bool = 0,
    zxindustry: bool = 0,
) -> pd.DataFrame:
    """专门计算因子值在各个申万一级行业上的Rank IC值，并绘制柱状图

    Parameters
    ----------
    df : pd.DataFrame
        全市场的因子值，index是时间，columns是股票代码
    excel_name : str, optional
        用于保存各个行业Rank IC值的excel文件的名字, by default '行业rankic.xlsx'
    png_name : str, optional
        用于保存各个行业Rank IC值的柱状图的名字, by default '行业rankic图.png'
    swindustry : bool, optional
        选择使用申万一级行业, by default 0
    zxindustry : bool, optional
        选择使用中信一级行业, by default 0

    Returns
    -------
    pd.DataFrame
        行业名称与对应的Rank IC
    """
    vs = group_test_on_industry(df, swindustry=swindustry, zxindustry=zxindustry)
    rankics = vs[["RankIC"]].T
    rankics.to_excel(excel_name)
    rankics.plot(kind="bar")
    plt.show()
    plt.savefig(png_name)
    return rankics


def long_test_on_industry(
    df: pd.DataFrame,
    nums: list,
    pos: bool = 0,
    neg: bool = 0,
    save_stock_list: bool = 0,
    swindustry: bool = 0,
    zxindustry: bool = 0,
) -> list[dict]:
    """对每个申万/中信一级行业成分股，使用某因子挑选出最多头的n值股票，考察其超额收益绩效、每月超额收益、每月每个行业的多头名单

    Parameters
    ----------
    df : pd.DataFrame
        使用的因子，index为时间，columns为股票代码
    nums : list
        多头想选取的股票的数量，例如[3,4,5]
    pos : bool, optional
        因子方向为正，即Rank IC为正，则指定此处为True, by default 0
    neg : bool, optional
        因子方向为负，即Rank IC为负，则指定此处为False, by default 0
    save_stock_list : bool, optional
        是否保存每月每个行业的多头名单，会降低运行速度, by default 0
    swindustry : bool, optional
        在申万一级行业上测试, by default 0
    zxindusrty : bool, optional
        在中信一级行业上测试, by default 0
    Returns
    -------
    list[dict]
        超额收益绩效、每月超额收益、每月每个行业的多头名单

    Raises
    ------
    IOError
        pos和neg必须有一个为1，否则将报错
    """
    fac = decap_industry(df, monthly=True)

    if swindustry:
        industry_dummy = pd.read_feather(
            homeplace.daily_data_file + "申万行业2021版哑变量.feather"
        ).fillna(0)
        indus = read_swindustry_prices()
    else:
        industry_dummy = pd.read_feather(
            homeplace.daily_data_file + "中信一级行业哑变量名称版.feather"
        ).fillna(0)
        indus = read_zxindustry_prices()
    inds = list(industry_dummy.columns)
    ret_next = (
        read_daily(close=1).resample("M").last()
        / read_daily(open=1).resample("M").first()
        - 1
    )
    ages = read_daily(age=1).resample("M").last()
    ages = (ages >= 60) + 0
    ages = ages.replace(0, np.nan)
    ret_next = ret_next * ages
    ret_next_dummy = 1 - ret_next.isna()

    def save_ind(code, num):
        ind = industry_dummy[["date", "code", code]]
        ind = ind.pivot(index="date", columns="code", values=code)
        ind = ind.resample("M").last()
        ind = ind.replace(0, np.nan)
        fi = ind * fac
        fi = fi.dropna(how="all")
        fi = fi.shift(1)
        fi = fi * ret_next_dummy
        fi = fi.dropna(how="all")

        def sing(x):
            if neg:
                thr = x.nsmallest(num).iloc[-1]
            elif pos:
                thr = x.nlargest(num).iloc[-1]
            else:
                raise IOError("您需要指定一下因子方向🤒")
            x = (x <= thr) + 0
            return x

        fi = fi.T.apply(sing).T
        fi = fi.replace(0, np.nan)
        fi = fi * ret_next
        ret_long = fi.mean(axis=1)
        return ret_long

    ret_longs = {k: [] for k in nums}
    if is_notebook():
        for num in tqdm.tqdm_notebook(nums):
            for code in inds[2:]:
                df = save_ind(code, num).to_frame(code)
                ret_longs[num] = ret_longs[num] + [df]
    else:
        for num in tqdm.tqdm(nums):
            for code in inds[2:]:
                df = save_ind(code, num).to_frame(code)
                ret_longs[num] = ret_longs[num] + [df]

    indus = indus.resample("M").last().pct_change()

    if swindustry:
        coms = {
            k: indus_name(pd.concat(v, axis=1).dropna(how="all").T).T
            for k, v in ret_longs.items()
        }
        rets = {
            k: (v - indus_name(indus.T).T).dropna(how="all") for k, v in coms.items()
        }
    else:
        coms = {k: pd.concat(v, axis=1).dropna(how="all") for k, v in ret_longs.items()}
        rets = {k: (v - indus).dropna(how="all") for k, v in coms.items()}

    nets = {k: (v + 1).cumprod() for k, v in rets.items()}
    nets = {
        k: v.apply(lambda x: x.dropna() / x.dropna().iloc[0]) for k, v in nets.items()
    }

    def comments_on_twins(nets: pd.Series, rets: pd.Series) -> pd.Series:
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

    if swindustry:
        name = "申万"
    else:
        name = "中信"
    w = pd.ExcelWriter(f"各个{name}一级行业多头超额绩效.xlsx")

    def com_all(df1, df2, num):
        cs = []
        for ind in list(df1.columns):
            c = comments_on_twins(df2[ind].dropna(), df1[ind].dropna()).to_frame(ind)
            cs.append(c)
        res = pd.concat(cs, axis=1).T
        res.to_excel(w, sheet_name=str(num))
        return res

    coms_finals = {k: com_all(rets[k], nets[k], k) for k in rets.keys()}
    w.save()
    w.close()

    rets_save = {k: v.dropna() for k, v in rets.items() if k in nums}
    u = pd.ExcelWriter(f"各个{name}一级行业每月超额收益率.xlsx")
    for k, v in rets_save.items():
        v.to_excel(u, sheet_name=str(k))
    u.save()
    u.close()

    if save_stock_list:

        def save_ind_stocks(code, num):
            ind = industry_dummy[["date", "code", code]]
            ind = ind.pivot(index="date", columns="code", values=code)
            ind = ind.replace(0, np.nan)
            fi = ind * fac
            fi = fi.dropna(how="all")
            fi = fi.shift(1)
            fi = fi * ret_next_dummy
            fi = fi.dropna(how="all")

            def sing(x):
                if neg:
                    thr = x.nsmallest(num)
                elif pos:
                    thr = x.nlargest(num)
                else:
                    raise IOError("您需要指定一下因子方向🤒")
                return tuple(thr.index)

            fi = fi.T.apply(sing)
            return fi

        stocks_longs = {k: {} for k in nums}
        if is_notebook():
            for num in tqdm.tqdm_notebook(nums):
                for code in inds[2:]:
                    stocks_longs[num][code] = save_ind_stocks(code, num)
        else:
            for num in tqdm.tqdm(nums):
                for code in inds[2:]:
                    stocks_longs[num][code] = save_ind_stocks(code, num)

        for num in nums:
            w1 = pd.ExcelWriter(f"各个{name}一级行业买{num}只的股票名单.xlsx")
            for k, v in stocks_longs[num].items():
                v = v.T
                v.index = v.index.strftime("%Y/%m/%d")
                v.to_excel(w1, sheet_name=INDUS_DICT[k])
            w1.save()
            w1.close()

        return [coms_finals, rets_save, stocks_longs]
    else:
        return [coms_finals, rets_save]


def long_test_on_swindustry(
    df: pd.DataFrame,
    nums: list,
    pos: bool = 0,
    neg: bool = 0,
    save_stock_list: bool = 0,
) -> list[dict]:
    """对每个申万一级行业成分股，使用某因子挑选出最多头的n值股票，考察其超额收益绩效、每月超额收益、每月每个行业的多头名单

    Parameters
    ----------
    df : pd.DataFrame
        使用的因子，index为时间，columns为股票代码
    nums : list
        多头想选取的股票的数量，例如[3,4,5]
    pos : bool, optional
        因子方向为正，即Rank IC为正，则指定此处为True, by default 0
    neg : bool, optional
        因子方向为负，即Rank IC为负，则指定此处为False, by default 0
    save_stock_list : bool, optional
        是否保存每月每个行业的多头名单，会降低运行速度, by default 0
    Returns
    -------
    list[dict]
        超额收益绩效、每月超额收益、每月每个行业的多头名单

    Raises
    ------
    IOError
        pos和neg必须有一个为1，否则将报错
    """
    res = long_test_on_industry(
        df=df,
        nums=nums,
        pos=pos,
        neg=neg,
        save_stock_list=save_stock_list,
        swindustry=1,
    )
    return res


def long_test_on_zxindustry(
    df: pd.DataFrame,
    nums: list,
    pos: bool = 0,
    neg: bool = 0,
    save_stock_list: bool = 0,
) -> list[dict]:
    """对每个中信一级行业成分股，使用某因子挑选出最多头的n值股票，考察其超额收益绩效、每月超额收益、每月每个行业的多头名单

    Parameters
    ----------
    df : pd.DataFrame
        使用的因子，index为时间，columns为股票代码
    nums : list
        多头想选取的股票的数量，例如[3,4,5]
    pos : bool, optional
        因子方向为正，即Rank IC为正，则指定此处为True, by default 0
    neg : bool, optional
        因子方向为负，即Rank IC为负，则指定此处为False, by default 0
    save_stock_list : bool, optional
        是否保存每月每个行业的多头名单，会降低运行速度, by default 0
    Returns
    -------
    list[dict]
        超额收益绩效、每月超额收益、每月每个行业的多头名单

    Raises
    ------
    IOError
        pos和neg必须有一个为1，否则将报错
    """
    res = long_test_on_industry(
        df=df,
        nums=nums,
        pos=pos,
        neg=neg,
        save_stock_list=save_stock_list,
        zxindustry=1,
    )
    return res


@kk.desktop_sender(title="嘿，行业中性化做完啦～🛁")
def decap(df: pd.DataFrame, daily: bool = 0, monthly: bool = 0) -> pd.DataFrame:
    """对因子做市值中性化

    Parameters
    ----------
    df : pd.DataFrame
        未中性化的因子，index是时间，columns是股票代码
    daily : bool, optional
        未中性化因子是日频的则为1，否则为0, by default 0
    monthly : bool, optional
        未中性化因子是月频的则为1，否则为0, by default 0

    Returns
    -------
    `pd.DataFrame`
        市值中性化之后的因子

    Raises
    ------
    `NotImplementedError`
        如果未指定日频或月频，将报错
    """
    if is_notebook():
        tqdm.tqdm_notebook().pandas()
    else:
        tqdm.tqdm.pandas()
    share = read_daily(sharenum=1)
    undi_close = read_daily(close=1, unadjust=1)
    cap = (share * undi_close).stack().reset_index()
    cap.columns = ["date", "code", "cap"]
    cap.cap = ss.boxcox(cap.cap)[0]

    def single(x):
        x.cap = ss.boxcox(x.cap)[0]
        return x

    cap = cap.groupby(["date"]).apply(single)
    cap = cap.set_index(["date", "code"]).unstack()
    cap.columns = [i[1] for i in list(cap.columns)]
    cap_monthly = cap.resample("M").last()
    last = df.resample("M").last()
    if df.shape[0] / last.shape[0] < 2:
        monthly = True
    else:
        daily = True
    if daily:
        df = (pure_fallmount(df) - (pure_fallmount(cap),))()
    elif monthly:
        df = (pure_fallmount(df) - (pure_fallmount(cap_monthly),))()
    else:
        raise NotImplementedError("必须指定频率")
    return df


@kk.desktop_sender(title="嘿，行业市值中性化做完啦～🛁")
def decap_industry(
    df: pd.DataFrame,
    daily: bool = 0,
    monthly: bool = 0,
    swindustry: bool = 0,
    zxindustry: bool = 0,
) -> pd.DataFrame:
    """对因子做行业市值中性化

    Parameters
    ----------
    df : pd.DataFrame
        未中性化的因子，index是时间，columns是股票代码
    daily : bool, optional
        未中性化因子是日频的则为1，否则为0, by default 0
    monthly : bool, optional
        未中性化因子是月频的则为1，否则为0, by default 0
    swindustry : bool, optional
        选择申万一级行业, by default 0
    zxindustry : bool, optional
        选择中信一级行业, by default 0

    Returns
    -------
    `pd.DataFrame`
        行业市值中性化之后的因子

    Raises
    ------
    `NotImplementedError`
        如果未指定日频或月频，将报错
    """
    start_date = int(datetime.datetime.strftime(df.index.min(), "%Y%m%d"))
    last = df.resample("M").last()
    homeplace = HomePlace()
    if daily == 0 and monthly == 0:
        if df.shape[0] / last.shape[0] < 2:
            monthly = True
        else:
            daily = True
    if monthly:
        cap = read_daily(flow_cap=1, start=start_date).resample("M").last()
    else:
        cap = read_daily(flow_cap=1, start=start_date)
    cap = cap.stack().reset_index()
    cap.columns = ["date", "code", "cap"]
    cap.cap = ss.boxcox(cap.cap)[0]

    def single(x):
        x.cap = ss.boxcox(x.cap)[0]
        return x

    cap = cap.groupby(["date"]).apply(single)
    df = df.stack().reset_index()
    df.columns = ["date", "code", "fac"]
    df = pd.merge(df, cap, on=["date", "code"])

    def neutralize_factors(df):
        """组内对因子进行市值中性化"""
        industry_codes = list(df.columns)
        industry_codes = [i for i in industry_codes if i.startswith("w")]
        industry_codes_str = "+".join(industry_codes)
        ols_result = smf.ols("fac~cap+" + industry_codes_str, data=df).fit()
        ols_w = ols_result.params["cap"]
        ols_b = ols_result.params["Intercept"]
        ols_bs = {}
        for ind in industry_codes:
            ols_bs[ind] = ols_result.params[ind]
        df.fac = df.fac - ols_w * df.cap - ols_b
        for k, v in ols_bs.items():
            df.fac = df.fac - v * df[k]
        df = df[["fac"]]
        return df

    if swindustry:
        file_name = "申万行业2021版哑变量.feather"
    else:
        file_name = "中信一级行业哑变量代码版.feather"

    if monthly:
        industry_dummy = (
            pd.read_feather(homeplace.daily_data_file + file_name)
            .fillna(0)
            .set_index("date")
            .groupby("code")
            .resample("M")
            .last()
        )
        industry_dummy = industry_dummy.fillna(0).drop(columns=["code"]).reset_index()
        industry_ws = [f"w{i}" for i in range(1, industry_dummy.shape[1] - 1)]
        col = ["code", "date"] + industry_ws
    elif daily:
        industry_dummy = pd.read_feather(homeplace.daily_data_file + file_name).fillna(
            0
        )
        industry_ws = [f"w{i}" for i in range(1, industry_dummy.shape[1] - 1)]
        col = ["date", "code"] + industry_ws
    else:
        raise NotImplementedError("必须指定频率")
    industry_dummy.columns = col
    df = pd.merge(df, industry_dummy, on=["date", "code"])
    df = df.set_index(["date", "code"])
    if is_notebook():
        tqdm.tqdm_notebook().pandas()
    else:
        tqdm.tqdm.pandas()
    df = df.groupby(["date"]).progress_apply(neutralize_factors)
    df = df.unstack()
    df.columns = [i[1] for i in list(df.columns)]
    return df


def deboth(df: pd.DataFrame) -> pd.DataFrame:
    """通过回测的方式，对月频因子做行业市值中性化

    Parameters
    ----------
    df : pd.DataFrame
        未中性化的因子

    Returns
    -------
    `pd.DataFrame`
        行业市值中性化之后的因子
    """
    shen = pure_moonnight(df, boxcox=1, plt_plot=0, print_comments=0)
    return shen()


def boom_four(
    df: pd.DataFrame, backsee: int = 20, daily: bool = 0, min_periods: int = None
) -> tuple[pd.DataFrame]:
    """生成20天均值，20天标准差，及二者正向z-score合成，正向排序合成，负向z-score合成，负向排序合成这6个因子

    Parameters
    ----------
    df : pd.DataFrame
        原日频因子
    backsee : int, optional
        回看天数, by default 20
    daily : bool, optional
        为1是每天都滚动，为0则仅保留月底值, by default 0
    min_periods : int, optional
        rolling时的最小期, by default backsee的一半

    Returns
    -------
    `tuple[pd.DataFrame]`
        6个因子的元组
    """
    if min_periods is None:
        min_periods = int(backsee * 0.5)
    if not daily:
        df_mean = (
            df.rolling(backsee, min_periods=min_periods).mean().resample("M").last()
        )
        df_std = df.rolling(backsee, min_periods=min_periods).std().resample("M").last()
        twins_add = (pure_fallmount(df_mean) + (pure_fallmount(df_std),))()
        rtwins_add = df_mean.rank(axis=1) + df_std.rank(axis=1)
        twins_minus = (pure_fallmount(df_mean) + (pure_fallmount(-df_std),))()
        rtwins_minus = df_mean.rank(axis=1) - df_std.rank(axis=1)
    else:
        df_mean = df.rolling(backsee, min_periods=min_periods).mean()
        df_std = df.rolling(backsee, min_periods=min_periods).std()
        twins_add = (pure_fallmount(df_mean) + (pure_fallmount(df_std),))()
        rtwins_add = df_mean.rank(axis=1) + df_std.rank(axis=1)
        twins_minus = (pure_fallmount(df_mean) + (pure_fallmount(-df_std),))()
        rtwins_minus = df_mean.rank(axis=1) - df_std.rank(axis=1)
    return df_mean, df_std, twins_add, rtwins_add, twins_minus, rtwins_minus


def add_cross_standardlize(*args: list) -> pd.DataFrame:
    """将众多因子横截面做z-score标准化之后相加

    Returns
    -------
    `pd.DataFrame`
        合成后的因子
    """
    fms = [pure_fallmount(i) for i in args]
    one = fms[0]
    others = fms[1:]
    final = one + others
    return final()


def to_tradeends(df: pd.DataFrame) -> pd.DataFrame:
    """将最后一个自然日改变为最后一个交易日

    Parameters
    ----------
    df : pd.DataFrame
        index为时间，为每个月的最后一天

    Returns
    -------
    `pd.DataFrame`
        修改为交易日标注后的pd.DataFrame
    """
    """"""
    trs = read_daily(tr=1)
    trs = trs.assign(tradeends=list(trs.index))
    trs = trs[["tradeends"]]
    trs = trs.resample("M").last()
    df = pd.concat([trs, df], axis=1)
    df = df.set_index(["tradeends"])
    return df


def market_kind(
    df: pd.DataFrame,
    zhuban: bool = 0,
    chuangye: bool = 0,
    kechuang: bool = 0,
    beijing: bool = 0,
) -> pd.DataFrame:
    """与宽基指数成分股的函数类似，限定股票在某个具体板块上

    Parameters
    ----------
    df : pd.DataFrame
        原始全部股票的因子值
    zhuban : bool, optional
        限定在主板范围内, by default 0
    chuangye : bool, optional
        限定在创业板范围内, by default 0
    kechuang : bool, optional
        限定在科创板范围内, by default 0
    beijing : bool, optional
        限定在北交所范围内, by default 0

    Returns
    -------
    `pd.DataFrame`
        限制范围后的因子值，其余为空

    Raises
    ------
    `ValueError`
        如果未指定任何股票池，将报错
    """
    trs = read_daily(tr=1)
    codes = list(trs.columns)
    dates = list(trs.index)
    if chuangye and kechuang:
        dummys = [1 if code[:2] in ["30", "68"] else np.nan for code in codes]
    else:
        if zhuban:
            dummys = [1 if code[:2] in ["00", "60"] else np.nan for code in codes]
        elif chuangye:
            dummys = [1 if code.startswith("3") else np.nan for code in codes]
        elif kechuang:
            dummys = [1 if code.startswith("68") else np.nan for code in codes]
        elif beijing:
            dummys = [1 if code.startswith("8") else np.nan for code in codes]
        else:
            raise ValueError("你总得选一个股票池吧？🤒")
    dummy_dict = {k: v for k, v in zip(codes, dummys)}
    dummy_df = pd.DataFrame(dummy_dict, index=dates)
    df = df * dummy_df
    return df


def show_corr(
    fac1: pd.DataFrame,
    fac2: pd.DataFrame,
    method: str = "spearman",
    plt_plot: bool = 1,
    show_series: bool = 0,
) -> float:
    """展示两个因子的截面相关性

    Parameters
    ----------
    fac1 : pd.DataFrame
        因子1
    fac2 : pd.DataFrame
        因子2
    method : str, optional
        计算相关系数的方法, by default "spearman"
    plt_plot : bool, optional
        是否画出相关系数的时序变化图, by default 1
    show_series : bool, optional
        返回相关性的序列，而非均值

    Returns
    -------
    `float`
        平均截面相关系数
    """
    corr = show_x_with_func(fac1, fac2, lambda x: x.corr(method=method).iloc[0, 1])
    if show_series:
        return corr
    else:
        if plt_plot:
            corr.plot(rot=60)
            plt.show()
        return corr.mean()


def show_corrs(
    factors: list[pd.DataFrame],
    factor_names: list[str] = None,
    print_bool: bool = True,
    show_percent: bool = True,
    method: str = "spearman",
) -> pd.DataFrame:
    """展示很多因子两两之间的截面相关性

    Parameters
    ----------
    factors : list[pd.DataFrame]
        所有因子构成的列表, by default None
    factor_names : list[str], optional
        上述因子依次的名字, by default None
    print_bool : bool, optional
        是否打印出两两之间相关系数的表格, by default True
    show_percent : bool, optional
        是否以百分数的形式展示, by default True
    method : str, optional
        计算相关系数的方法, by default "spearman"

    Returns
    -------
    `pd.DataFrame`
        两两之间相关系数的表格
    """
    corrs = []
    for i in range(len(factors)):
        main_i = factors[i]
        follows = factors[i + 1 :]
        corr = [show_corr(main_i, i, plt_plot=False, method=method) for i in follows]
        corr = [np.nan] * (i + 1) + corr
        corrs.append(corr)
    if factor_names is None:
        factor_names = [f"fac{i}" for i in list(range(1, len(factors) + 1))]
    corrs = pd.DataFrame(corrs, columns=factor_names, index=factor_names)
    np.fill_diagonal(corrs.to_numpy(), 1)
    if show_percent:
        pcorrs = corrs.applymap(to_percent)
    else:
        pcorrs = corrs.copy()
    if print_bool:
        print(pcorrs)
    return corrs


def show_cov(
    fac1: pd.DataFrame,
    fac2: pd.DataFrame,
    plt_plot: bool = 1,
    show_series: bool = 0,
) -> float:
    """展示两个因子的截面相关性

    Parameters
    ----------
    fac1 : pd.DataFrame
        因子1
    fac2 : pd.DataFrame
        因子2
    method : str, optional
        计算相关系数的方法, by default "spearman"
    plt_plot : bool, optional
        是否画出相关系数的时序变化图, by default 1
    show_series : bool, optional
        返回相关性的序列，而非均值

    Returns
    -------
    `float`
        平均截面相关系数
    """
    cov = show_x_with_func(fac1, fac2, lambda x: x.cov().iloc[0, 1])
    if show_series:
        return cov
    else:
        if plt_plot:
            cov.plot(rot=60)
            plt.show()
        return cov.mean()


def show_x_with_func(
    fac1: pd.DataFrame,
    fac2: pd.DataFrame,
    func: Callable,
) -> pd.Series:
    """展示两个因子的某种截面关系

    Parameters
    ----------
    fac1 : pd.DataFrame
        因子1
    fac2 : pd.DataFrame
        因子2
    func : Callable
        要对两个因子在截面上的进行的操作

    Returns
    -------
    `pd.Series`
        截面关系
    """
    the_func = partial(func)
    both1 = fac1.stack().reset_index()
    befo1 = fac2.stack().reset_index()
    both1.columns = ["date", "code", "both"]
    befo1.columns = ["date", "code", "befo"]
    twins = pd.merge(both1, befo1, on=["date", "code"]).set_index(["date", "code"])
    corr = twins.groupby("date").apply(the_func)
    return corr


def show_covs(
    factors: list[pd.DataFrame],
    factor_names: list[str] = None,
    print_bool: bool = True,
    show_percent: bool = True,
    method: str = "spearman",
) -> pd.DataFrame:
    """展示很多因子两两之间的截面相关性

    Parameters
    ----------
    factors : list[pd.DataFrame]
        所有因子构成的列表, by default None
    factor_names : list[str], optional
        上述因子依次的名字, by default None
    print_bool : bool, optional
        是否打印出两两之间相关系数的表格, by default True
    show_percent : bool, optional
        是否以百分数的形式展示, by default True
    method : str, optional
        计算相关系数的方法, by default "spearman"

    Returns
    -------
    `pd.DataFrame`
        两两之间相关系数的表格
    """
    corrs = []
    for i in range(len(factors)):
        main_i = factors[i]
        follows = factors[i + 1 :]
        corr = [show_cov(main_i, i, plt_plot=False) for i in follows]
        corr = [np.nan] * (i + 1) + corr
        corrs.append(corr)
    if factor_names is None:
        factor_names = [f"fac{i}" for i in list(range(1, len(factors) + 1))]
    corrs = pd.DataFrame(corrs, columns=factor_names, index=factor_names)
    np.fill_diagonal(corrs.to_numpy(), 1)
    if show_percent:
        pcorrs = corrs.applymap(to_percent)
    else:
        pcorrs = corrs.copy()
    if print_bool:
        print(pcorrs)
    return corrs


def de_cross(
    y: pd.DataFrame, xs: Union[list[pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    """使用若干因子对某个因子进行正交化处理

    Parameters
    ----------
    y : pd.DataFrame
        研究的目标，回归中的y
    xs : Union[list[pd.DataFrame],pd.DataFrame]
        用于正交化的若干因子，回归中的x

    Returns
    -------
    pd.DataFrame
        正交化之后的因子
    """
    if not isinstance(xs, list):
        xs = [xs]
    y = pure_fallmount(y)
    xs = [pure_fallmount(i) for i in xs]
    return (y - xs)()


def show_corrs_with_old(
    df: pd.DataFrame = None, method: str = "spearman"
) -> pd.DataFrame:
    """计算新因子和已有因子的相关系数

    Parameters
    ----------
    df : pd.DataFrame, optional
        新因子, by default None
    method : str, optional
        求相关系数的方法, by default 'spearman'

    Returns
    -------
    pd.DataFrame
        相关系数矩阵
    """
    if df is not None:
        df0 = df.resample("M").last()
        if df.shape[0] / df0.shape[0] > 2:
            daily = 1
        else:
            daily = 0
    olds = []
    for i in range(1, 100):
        try:
            if daily:
                old = database_read_final_factors(order=i)[0]
            else:
                old = database_read_final_factors(order=i)[0].resample("M").last()
            olds.append(old)
        except Exception:
            break
    if df is not None:
        olds = [df] + olds
        corrs = show_corrs(
            olds, ["new"] + [f"old{i}" for i in range(1, len(olds))], method=method
        )
    else:
        corrs = show_corrs(
            olds, [f"old{i}" for i in range(1, len(olds))], method=method
        )
    return corrs


class pure_moon(object):
    __slots__ = [
        "homeplace",
        "sts_monthly_file",
        "states_monthly_file",
        "factors",
        "codes",
        "tradedays",
        "ages",
        "amounts",
        "closes",
        "opens",
        "capital",
        "states",
        "sts",
        "turnovers",
        "sts_monthly",
        "states_monthly",
        "ages_monthly",
        "tris_monthly",
        "opens_monthly",
        "closes_monthly",
        "rets_monthly",
        "opens_monthly_shift",
        "rets_monthly_begin",
        "limit_ups",
        "limit_downs",
        "data",
        "ic_icir_and_rank",
        "rets_monthly_limit_downs",
        "group_rets",
        "long_short_rets",
        "long_short_net_values",
        "group_net_values",
        "long_short_ret_yearly",
        "long_short_vol_yearly",
        "long_short_info_ratio",
        "long_short_win_times",
        "long_short_win_ratio",
        "retreats",
        "max_retreat",
        "long_short_comments",
        "total_comments",
        "square_rets",
        "cap",
        "cap_value",
        "industry_dummy",
        "industry_codes",
        "industry_codes_str",
        "industry_ws",
        "__factors_out",
        "ics",
        "rankics",
        "factor_turnover_rates",
        "factor_turnover_rate",
        "group_rets_std",
        "group_rets_stds",
        "wind_out",
        "swindustry_dummy",
        "zxindustry_dummy",
        "closes2_monthly",
        "rets_monthly_last",
    ]

    @classmethod
    @lru_cache(maxsize=None)
    def __init__(
        cls,
        no_read_indu: bool = 0,
        swindustry_dummy: pd.DataFrame = None,
        zxindustry_dummy: pd.DataFrame = None,
        read_in_swindustry_dummy: bool = 0,
    ):
        cls.homeplace = HomePlace()
        # 已经算好的月度st状态文件
        cls.sts_monthly_file = homeplace.daily_data_file + "sts_monthly.feather"
        # 已经算好的月度交易状态文件
        cls.states_monthly_file = homeplace.daily_data_file + "states_monthly.feather"

        if swindustry_dummy is not None:
            cls.swindustry_dummy = swindustry_dummy
        if zxindustry_dummy is not None:
            cls.zxindustry_dummy = zxindustry_dummy

        def deal_dummy(industry_dummy):
            industry_dummy = industry_dummy.drop(columns=["code"]).reset_index()
            industry_ws = [f"w{i}" for i in range(1, industry_dummy.shape[1] - 1)]
            col = ["code", "date"] + industry_ws
            industry_dummy.columns = col
            industry_dummy = industry_dummy[
                industry_dummy.date >= pd.Timestamp("20100101")
            ]
            return industry_dummy

        if (swindustry_dummy is None) and (zxindustry_dummy is None):

            if not no_read_indu:
                if read_in_swindustry_dummy:
                    cls.swindustry_dummy = (
                        pd.read_feather(
                            cls.homeplace.daily_data_file + "申万行业2021版哑变量.feather"
                        )
                        .fillna(0)
                        .set_index("date")
                        .groupby("code")
                        .resample("M")
                        .last()
                    )
                    cls.swindustry_dummy = deal_dummy(cls.swindustry_dummy)

                cls.zxindustry_dummy = (
                    pd.read_feather(
                        cls.homeplace.daily_data_file + "中信一级行业哑变量代码版.feather"
                    )
                    .fillna(0)
                    .set_index("date")
                    .groupby("code")
                    .resample("M")
                    .last()
                    .fillna(0)
                )

                cls.zxindustry_dummy = deal_dummy(cls.zxindustry_dummy)

    @property
    def factors_out(self):
        return self.__factors_out

    def __call__(self):
        """调用对象则返回因子值"""
        return self.factors_out

    @classmethod
    def set_basic_data(
        cls,
        age: pd.DataFrame,
        st: pd.DataFrame,
        state: pd.DataFrame,
        open: pd.DataFrame,
        close: pd.DataFrame,
        capital: pd.DataFrame,
    ):
        # 上市天数文件
        cls.ages = age
        # st日子标志文件
        cls.sts = st.fillna(0)
        # cls.sts = 1 - cls.sts.fillna(0)
        # 交易状态文件
        cls.states = state
        # 复权开盘价数据文件
        cls.opens = open
        # 复权收盘价数据文件
        cls.closes = close
        # 月底流通市值数据
        cls.capital = capital
        cls.opens = cls.opens.replace(0, np.nan)
        cls.closes = cls.closes.replace(0, np.nan)

    def set_factor_df_date_as_index(self, df):
        """设置因子数据的dataframe，因子表列名应为股票代码，索引应为时间"""
        df = df.reset_index()
        df.columns = ["date"] + list(df.columns)[1:]
        self.factors = df
        self.factors = self.factors.set_index("date")
        self.factors = self.factors.resample("M").last()
        self.factors = self.factors.reset_index()

    @classmethod
    def judge_month_st(cls, df):
        """比较一个月内st的天数，如果st天数多，就删除本月，如果正常多，就保留本月"""
        st_count = len(df[df == 1])
        normal_count = len(df[df != 1])
        if st_count >= normal_count:
            return 0
        else:
            return 1

    @classmethod
    def judge_month_state(cls, df):
        """比较一个月内非正常交易的天数，如果非正常交易天数多，就删除本月，否则保留本月"""
        abnormal_count = len(df[df == 0])
        normal_count = len(df[df == 1])
        if abnormal_count >= normal_count:
            return 0
        else:
            return 1

    @classmethod
    def read_add(cls, pridf, df, func):
        """由于数据更新，过去计算的月度状态可能需要追加"""
        if pridf.index.max() > df.index.max():
            df_add = pridf[pridf.index > df.index.max()]
            df_add = df_add.resample("M").apply(func)
            df = pd.concat([df, df_add])
            return df
        else:
            return df

    @classmethod
    def daily_to_monthly(cls, pridf, path, func):
        """把日度的交易状态、st、上市天数，转化为月度的，并生成能否交易的判断
        读取本地已经算好的文件，并追加新的时间段部分，如果本地没有就直接全部重新算"""
        try:
            month_df = pd.read_feather(path)
            month_df = month_df.set_index(list(month_df.columns)[0])
            month_df = cls.read_add(pridf, month_df, func)
            month_df.reset_index().to_feather(path)
        except Exception as e:
            if not STATES["NO_LOG"]:
                logger.error("error occurs when read state files")
                logger.error(e)
            print("state file rewriting……")
            month_df = pridf.resample("M").apply(func)
            month_df.reset_index().to_feather(path)
        return month_df

    @classmethod
    @lru_cache(maxsize=None)
    def judge_month(cls):
        """生成一个月综合判断的表格"""

        cls.sts_monthly = cls.daily_to_monthly(
            cls.sts, cls.sts_monthly_file, cls.judge_month_st
        )
        cls.states_monthly = cls.daily_to_monthly(
            cls.states, cls.states_monthly_file, cls.judge_month_state
        )
        cls.ages_monthly = cls.ages.resample("M").last()
        cls.ages_monthly = np.sign(cls.ages_monthly.applymap(lambda x: x - 60)).replace(
            -1, 0
        )
        cls.tris_monthly = cls.sts_monthly * cls.states_monthly * cls.ages_monthly
        cls.tris_monthly = cls.tris_monthly.replace(0, np.nan)

    @classmethod
    @lru_cache(maxsize=None)
    def get_rets_month(cls):
        """计算每月的收益率，并根据每月做出交易状态，做出删减"""
        cls.opens_monthly = cls.opens.resample("M").first()
        cls.closes_monthly = cls.closes.resample("M").last()
        cls.rets_monthly = (cls.closes_monthly - cls.opens_monthly) / cls.opens_monthly
        cls.rets_monthly = cls.rets_monthly * cls.tris_monthly
        cls.rets_monthly = cls.rets_monthly.stack().reset_index()
        cls.rets_monthly.columns = ["date", "code", "ret"]

    @classmethod
    def neutralize_factors(cls, df):
        """组内对因子进行市值中性化"""
        industry_codes = list(df.columns)
        industry_codes = [i for i in industry_codes if i.startswith("w")]
        industry_codes_str = "+".join(industry_codes)
        if len(industry_codes_str) > 0:
            ols_result = smf.ols("fac~cap_size+" + industry_codes_str, data=df).fit()
        else:
            ols_result = smf.ols("fac~cap_size", data=df).fit()
        ols_w = ols_result.params["cap_size"]
        ols_b = ols_result.params["Intercept"]
        ols_bs = {}
        for ind in industry_codes:
            ols_bs[ind] = ols_result.params[ind]
        df.fac = df.fac - ols_w * df.cap_size - ols_b
        for k, v in ols_bs.items():
            df.fac = df.fac - v * df[k]
        df = df[["fac"]]
        return df

    @classmethod
    @lru_cache(maxsize=None)
    def get_log_cap(cls, boxcox=True):
        """获得对数市值"""
        cls.cap = cls.capital.stack().reset_index()
        cls.cap.columns = ["date", "code", "cap_size"]
        if boxcox:

            def single(x):
                x.cap_size = ss.boxcox(x.cap_size)[0]
                return x

            cls.cap = cls.cap.groupby(["date"]).apply(single)
        else:
            cls.cap["cap_size"] = np.log(cls.cap["cap_size"])

    def get_neutral_factors(
        self, zxindustry_dummies=0, swindustry_dummies=0, only_cap=0
    ):
        """对因子进行行业市值中性化"""
        self.factors = self.factors.set_index("date")
        self.factors.index = self.factors.index + pd.DateOffset(months=1)
        self.factors = self.factors.resample("M").last()
        last_date = self.tris_monthly.index.max() + pd.DateOffset(months=1)
        last_date = last_date + pd.tseries.offsets.MonthEnd()
        add_tail = pd.DataFrame(1, index=[last_date], columns=self.tris_monthly.columns)
        tris_monthly = pd.concat([self.tris_monthly, add_tail])
        self.factors = self.factors * tris_monthly
        self.factors.index = self.factors.index - pd.DateOffset(months=1)
        self.factors = self.factors.resample("M").last()
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ["date", "code", "fac"]
        self.factors = pd.merge(
            self.factors, self.cap, how="inner", on=["date", "code"]
        )
        if not only_cap:
            if swindustry_dummies:
                self.factors = pd.merge(
                    self.factors, self.swindustry_dummy, on=["date", "code"]
                )
            else:
                self.factors = pd.merge(
                    self.factors, self.zxindustry_dummy, on=["date", "code"]
                )
        self.factors = self.factors.set_index(["date", "code"])
        self.factors = self.factors.groupby(["date"]).apply(self.neutralize_factors)
        self.factors = self.factors.reset_index()

    def deal_with_factors(self):
        """删除不符合交易条件的因子数据"""
        self.factors = self.factors.set_index("date")
        self.__factors_out = self.factors.copy()
        self.__factors_out.columns = [i[1] for i in list(self.__factors_out.columns)]
        self.factors.index = self.factors.index + pd.DateOffset(months=1)
        self.factors = self.factors.resample("M").last()
        self.factors = self.factors * self.tris_monthly
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ["date", "code", "fac"]

    def deal_with_factors_after_neutralize(self):
        """中性化之后的因子处理方法"""
        self.factors = self.factors.set_index(["date", "code"])
        self.factors = self.factors.unstack()
        self.__factors_out = self.factors.copy()
        self.__factors_out.columns = [i[1] for i in list(self.__factors_out.columns)]
        self.factors.index = self.factors.index + pd.DateOffset(months=1)
        self.factors = self.factors.resample("M").last()
        self.factors.columns = list(map(lambda x: x[1], list(self.factors.columns)))
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ["date", "code", "fac"]

    @classmethod
    def find_limit(cls, df, up=1):
        """计算涨跌幅超过9.8%的股票，并将其存储进一个长列表里
        其中时间列，为某月的最后一天；涨停日虽然为下月初第一天，但这里标注的时间统一为上月最后一天"""
        limit_df = np.sign(df.applymap(lambda x: x - up * 0.098)).replace(
            -1 * up, np.nan
        )
        limit_df = limit_df.stack().reset_index()
        limit_df.columns = ["date", "code", "limit_up_signal"]
        limit_df = limit_df[["date", "code"]]
        return limit_df

    @classmethod
    @lru_cache(maxsize=None)
    def get_limit_ups_downs(cls):
        """找月初第一天就涨停"""
        """或者是月末跌停的股票"""
        cls.opens_monthly_shift = cls.opens_monthly.copy()
        cls.opens_monthly_shift = cls.opens_monthly_shift.shift(-1)
        cls.rets_monthly_begin = (
            cls.opens_monthly_shift - cls.closes_monthly
        ) / cls.closes_monthly
        cls.closes2_monthly = cls.closes.shift(1).resample("M").last()
        cls.rets_monthly_last = (
            cls.closes_monthly - cls.closes2_monthly
        ) / cls.closes2_monthly
        cls.limit_ups = cls.find_limit(cls.rets_monthly_begin, up=1)
        cls.limit_downs = cls.find_limit(cls.rets_monthly_last, up=-1)

    def get_ic_rankic(cls, df):
        """计算IC和RankIC"""
        df1 = df[["ret", "fac"]]
        ic = df1.corr(method="pearson").iloc[0, 1]
        rankic = df1.corr(method="spearman").iloc[0, 1]
        df2 = pd.DataFrame({"ic": [ic], "rankic": [rankic]})
        return df2

    def get_icir_rankicir(cls, df):
        """计算ICIR和RankICIR"""
        ic = df.ic.mean()
        rankic = df.rankic.mean()
        icir = ic / np.std(df.ic) * (12 ** (0.5))
        rankicir = rankic / np.std(df.rankic) * (12 ** (0.5))
        return pd.DataFrame(
            {"IC": [ic], "ICIR": [icir], "RankIC": [rankic], "RankICIR": [rankicir]},
            index=["评价指标"],
        )

    def get_ic_icir_and_rank(cls, df):
        """计算IC、ICIR、RankIC、RankICIR"""
        df1 = df.groupby("date").apply(cls.get_ic_rankic)
        cls.ics = df1.ic
        cls.rankics = df1.rankic
        cls.ics = cls.ics.reset_index(drop=True, level=1).to_frame()
        cls.rankics = cls.rankics.reset_index(drop=True, level=1).to_frame()
        df2 = cls.get_icir_rankicir(df1)
        df2 = df2.T
        dura = (df.date.max() - df.date.min()).days / 365
        t_value = df2.iloc[3, 0] * (dura ** (1 / 2))
        df3 = pd.DataFrame({"评价指标": [t_value]}, index=["RankIC均值t值"])
        df4 = pd.concat([df2, df3])
        return df4

    @classmethod
    def get_groups(cls, df, groups_num):
        """依据因子值，判断是在第几组"""
        if "group" in list(df.columns):
            df = df.drop(columns=["group"])
        df = df.sort_values(["fac"], ascending=True)
        each_group = round(df.shape[0] / groups_num)
        l = list(
            map(
                lambda x, y: [x] * y,
                list(range(1, groups_num + 1)),
                [each_group] * groups_num,
            )
        )
        l = reduce(lambda x, y: x + y, l)
        if len(l) < df.shape[0]:
            l = l + [groups_num] * (df.shape[0] - len(l))
        l = l[: df.shape[0]]
        df.insert(0, "group", l)
        return df

    @classmethod
    def next_month_end(cls, x):
        """找到下个月最后一天"""
        x1 = x = x + relativedelta(months=1)
        while x1.month == x.month:
            x1 = x1 + relativedelta(days=1)
        return x1 - relativedelta(days=1)

    @classmethod
    def limit_old_to_new(cls, limit, data):
        """获取跌停股在旧月的组号，然后将日期调整到新月里
        涨停股则获得新月里涨停股的代码和时间，然后直接删去"""
        data1 = data.copy()
        data1 = data1.reset_index()
        data1.columns = ["data_index"] + list(data1.columns)[1:]
        old = pd.merge(limit, data1, how="inner", on=["date", "code"])
        old = old.set_index("data_index")
        old = old[["group", "date", "code"]]
        old.date = list(map(cls.next_month_end, list(old.date)))
        return old

    def get_data(self, groups_num):
        """拼接因子数据和每月收益率数据，并对涨停和跌停股加以处理"""
        self.data = pd.merge(
            self.rets_monthly, self.factors, how="inner", on=["date", "code"]
        )
        self.ic_icir_and_rank = self.get_ic_icir_and_rank(self.data)
        self.data = self.data.groupby("date").apply(
            lambda x: self.get_groups(x, groups_num)
        )
        self.wind_out = self.data.copy()
        self.factor_turnover_rates = self.data.pivot(
            index="date", columns="code", values="group"
        )
        self.factor_turnover_rates = self.factor_turnover_rates.diff()
        change = ((np.abs(np.sign(self.factor_turnover_rates)) == 1) + 0).sum(axis=1)
        still = ((self.factor_turnover_rates == 0) + 0).sum(axis=1)
        self.factor_turnover_rates = change / (change + still)
        self.factor_turnover_rate = self.factor_turnover_rates.mean()
        self.data = self.data.reset_index(drop=True)
        limit_ups_object = self.limit_old_to_new(self.limit_ups, self.data)
        limit_downs_object = self.limit_old_to_new(self.limit_downs, self.data)
        self.data = self.data.drop(limit_ups_object.index)
        rets_monthly_limit_downs = pd.merge(
            self.rets_monthly, limit_downs_object, how="inner", on=["date", "code"]
        )
        self.data = pd.concat([self.data, rets_monthly_limit_downs])

    def select_data_time(self, time_start, time_end):
        """筛选特定的时间段"""
        if time_start:
            self.data = self.data[self.data.date >= time_start]
        if time_end:
            self.data = self.data[self.data.date <= time_end]

    def make_start_to_one(self, l):
        """让净值序列的第一个数变成1"""
        min_date = self.factors.date.min()
        add_date = min_date - relativedelta(days=min_date.day)
        add_l = pd.Series([1], index=[add_date])
        l = pd.concat([add_l, l])
        return l

    def to_group_ret(self, l):
        """每一组的年化收益率"""
        ret = l[-1] ** (12 / len(l)) - 1
        return ret

    def get_group_rets_net_values(self, groups_num=10, value_weighted=False):
        """计算组内每一期的平均收益，生成每日收益率序列和净值序列"""
        if value_weighted:
            cap_value = self.capital.copy()
            cap_value = cap_value.resample("M").last().shift(1)
            cap_value = cap_value * self.tris_monthly
            # cap_value=np.log(cap_value)
            cap_value = cap_value.stack().reset_index()
            cap_value.columns = ["date", "code", "cap_value"]
            self.data = pd.merge(self.data, cap_value, on=["date", "code"])

            def in_g(df):
                df.cap_value = df.cap_value / df.cap_value.sum()
                df.ret = df.ret * df.cap_value
                return df.ret.sum()

            self.group_rets = self.data.groupby(["date", "group"]).apply(in_g)
            self.group_rets_std = "市值加权暂未设置该功能，敬请期待🌙"
        else:
            self.group_rets = self.data.groupby(["date", "group"]).apply(
                lambda x: x.ret.mean()
            )
            self.group_rets_stds = self.data.groupby(["date", "group"]).apply(
                lambda x: x.ret.std()
            )
            self.group_rets_std = self.group_rets_stds.groupby("group").mean()
        # dropna是因为如果股票行情数据比因子数据的截止日期晚，而最后一个月发生月初跌停时，会造成最后某组多出一个月的数据
        self.group_rets = self.group_rets.unstack()
        self.group_rets = self.group_rets[
            self.group_rets.index <= self.factors.date.max()
        ]
        self.group_rets.columns = list(map(str, list(self.group_rets.columns)))
        self.group_rets = self.group_rets.add_prefix("group")
        self.long_short_rets = (
            self.group_rets["group1"] - self.group_rets["group" + str(groups_num)]
        )
        self.long_short_net_values = (self.long_short_rets + 1).cumprod()
        if self.long_short_net_values[-1] <= self.long_short_net_values[0]:
            self.long_short_rets = (
                self.group_rets["group" + str(groups_num)] - self.group_rets["group1"]
            )
            self.long_short_net_values = (self.long_short_rets + 1).cumprod()
        self.long_short_net_values = self.make_start_to_one(self.long_short_net_values)
        self.group_rets = self.group_rets.assign(long_short=self.long_short_rets)
        self.group_net_values = self.group_rets.applymap(lambda x: x + 1)
        self.group_net_values = self.group_net_values.cumprod()
        self.group_net_values = self.group_net_values.apply(self.make_start_to_one)
        a = groups_num ** (0.5)
        # 判断是否要两个因子画表格
        if a == int(a):
            self.square_rets = (
                self.group_net_values.iloc[:, :-1].apply(self.to_group_ret).to_numpy()
            )
            self.square_rets = self.square_rets.reshape((int(a), int(a)))
            self.square_rets = pd.DataFrame(
                self.square_rets,
                columns=list(range(1, int(a) + 1)),
                index=list(range(1, int(a) + 1)),
            )
            print("这是self.square_rets", self.square_rets)

    def get_long_short_comments(self, on_paper=False):
        """计算多空对冲的相关评价指标
        包括年化收益率、年化波动率、信息比率、月度胜率、最大回撤率"""
        self.long_short_ret_yearly = (
            self.long_short_net_values[-1] ** (12 / len(self.long_short_net_values)) - 1
        )
        self.long_short_vol_yearly = np.std(self.long_short_rets) * (12**0.5)
        self.long_short_info_ratio = (
            self.long_short_ret_yearly / self.long_short_vol_yearly
        )
        self.long_short_win_times = len(self.long_short_rets[self.long_short_rets > 0])
        self.long_short_win_ratio = self.long_short_win_times / len(
            self.long_short_rets
        )
        self.max_retreat = -(
            self.long_short_net_values / self.long_short_net_values.expanding(1).max()
            - 1
        ).min()
        if on_paper:
            self.long_short_comments = pd.DataFrame(
                {
                    "评价指标": [
                        self.long_short_ret_yearly,
                        self.long_short_vol_yearly,
                        self.long_short_info_ratio,
                        self.long_short_win_ratio,
                        self.max_retreat,
                    ]
                },
                index=["年化收益率", "年化波动率", "收益波动比", "月度胜率", "最大回撤率"],
            )
        else:
            self.long_short_comments = pd.DataFrame(
                {
                    "评价指标": [
                        self.long_short_ret_yearly,
                        self.long_short_vol_yearly,
                        self.long_short_info_ratio,
                        self.long_short_win_ratio,
                        self.max_retreat,
                    ]
                },
                index=["年化收益率", "年化波动率", "信息比率", "月度胜率", "最大回撤率"],
            )

    def get_total_comments(self):
        """综合IC、ICIR、RankIC、RankICIR,年化收益率、年化波动率、信息比率、月度胜率、最大回撤率"""
        self.total_comments = pd.concat(
            [
                self.ic_icir_and_rank,
                self.long_short_comments,
                pd.DataFrame({"评价指标": [self.factor_turnover_rate]}, index=["月均换手率"]),
            ]
        )

    def plot_net_values(self, y2, filename, iplot=1, ilegend=1):
        """使用matplotlib来画图，y2为是否对多空组合采用双y轴"""
        if not iplot:
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(33, 8))
            self.group_net_values.plot(secondary_y=y2, rot=60, ax=ax[0])
            self.group_net_values.plot(secondary_y=y2, ax=ax[0])
            b = self.rankics.copy()
            b.index = [int(i.year) if i.month == 1 else "" for i in list(b.index)]
            b.plot(kind="bar", rot=60, ax=ax[1])
            self.factor_turnover_rates.plot(rot=60, ax=ax[2])

            filename_path = filename + ".png"
            if not STATES["NO_SAVE"]:
                plt.savefig(filename_path)
        else:
            tris = pd.concat(
                [self.group_net_values, self.factor_turnover_rates, self.rankics],
                axis=1,
            ).rename(columns={0: "turnover_rate"})
            figs = cf.figures(
                tris,
                [
                    dict(kind="line", y=list(self.group_net_values.columns)),
                    dict(kind="line", y="turnover_rate"),
                    dict(kind="bar", y="rankic"),
                ],
                asList=True,
            )
            comments = (
                self.total_comments.applymap(lambda x: round(x, 4))
                .rename(index={"RankIC均值t值": "RankIC.t"})
                .reset_index()
            )
            here = pd.concat(
                [
                    comments.iloc[:5, :].reset_index(drop=True),
                    comments.iloc[5:, :].reset_index(drop=True),
                ],
                axis=1,
            )
            here.columns = ["信息系数", "结果", "绩效指标", "结果"]
            # here=here.to_numpy().tolist()+[['信息系数','结果','绩效指标','结果']]
            table = FF.create_table(here.iloc[::-1])
            table.update_yaxes(matches=None)
            # table=go.Figure([go.Table(header=dict(values=list(here.columns)),cells=dict(values=here.to_numpy().tolist()))])
            figs.append(table)
            figs = [figs[-1]] + figs[:-1]
            figs[1].update_layout(
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            base_layout = cf.tools.get_base_layout(figs)

            sp = cf.subplots(
                figs,
                shape=(2, 10),
                base_layout=base_layout,
                vertical_spacing=0.15,
                horizontal_spacing=0.03,
                shared_yaxes=False,
                specs=[
                    [
                        {"rowspan": 2, "colspan": 3},
                        None,
                        None,
                        {"rowspan": 2, "colspan": 4},
                        None,
                        None,
                        None,
                        {"colspan": 3},
                        None,
                        None,
                    ],
                    [
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        {"colspan": 3},
                        None,
                        None,
                    ],
                ],
                subplot_titles=["净值曲线", "月换手率", "Rank IC时序图", "绩效指标"],
            )
            sp["layout"].update(showlegend=ilegend)
            # los=sp['layout']['annotations']
            # los[0]['font']['color']='#000000'
            # los[1]['font']['color']='#000000'
            # los[2]['font']['color']='#000000'
            # los[3]['font']['color']='#000000'
            # los[-1]['font']['color']='#ffffff'
            # los[-2]['font']['color']='#ffffff'
            # los[-3]['font']['color']='#ffffff'
            # los[-4]['font']['color']='#ffffff'
            # los[0]['text']=los[0]['text'][3:-4]
            # los[1]['text']=los[1]['text'][3:-4]
            # los[2]['text']=los[2]['text'][3:-4]
            # los[3]['text']=los[3]['text'][3:-4]
            # los[-1]['text']='<b>'+los[-1]['text']+'</b>'
            # los[-2]['text']='<b>'+los[-2]['text']+'</b>'
            # los[-3]['text']='<b>'+los[-3]['text']+'</b>'
            # los[-4]['text']='<b>'+los[-4]['text']+'</b>'
            # sp['layout']['annotations']=los
            # print(sp['layout']['annotations'])
            # sp['layout']['annotations'][0]['yanchor']='top'
            cf.iplot(sp)
            # tris=pd.concat([self.group_net_values,self.rankics,self.factor_turnover_rates],axis=1).rename(columns={0:'turnover_rate'})
            # sp=plyoo.make_subplots(rows=2,cols=8,vertical_spacing=.15,horizontal_spacing=.03,
            #                specs=[[{'rowspan':2,'colspan':2,'type':'domain'},None,{'rowspan':2,'colspan':4,'type':'xy'},None,None,None,{'colspan':2,'type':'xy'},None],
            #                       [None,None,None,None,None,None,{'colspan':2,'type':'xy'},None]],
            #                subplot_titles=['净值曲线','Rank IC时序图','月换手率','绩效指标'])
            # comments=self.total_comments.applymap(lambda x:round(x,4)).rename(index={'RankIC均值t值':'RankIC.t'}).reset_index()
            # here=pd.concat([comments.iloc[:5,:].reset_index(drop=True),comments.iloc[5:,:].reset_index(drop=True)],axis=1)
            # here.columns=['信息系数','结果','绩效指标','结果']
            # table=FF.create_table(here)
            # sp.add_trace(table)

    def plotly_net_values(self, filename):
        """使用plotly.express画图"""
        fig = pe.line(self.group_net_values)
        filename_path = filename + ".html"
        pio.write_html(fig, filename_path, auto_open=True)

    @classmethod
    @lru_cache(maxsize=None)
    def prerpare(cls):
        """通用数据准备"""
        cls.judge_month()
        cls.get_rets_month()

    @kk.desktop_sender(title="嘿，回测结束啦～🗓")
    def run(
        self,
        groups_num=10,
        neutralize=False,
        boxcox=False,
        value_weighted=False,
        y2=False,
        plt_plot=True,
        plotly_plot=False,
        filename="分组净值图",
        time_start=None,
        time_end=None,
        print_comments=True,
        comments_writer=None,
        net_values_writer=None,
        rets_writer=None,
        comments_sheetname=None,
        net_values_sheetname=None,
        rets_sheetname=None,
        on_paper=False,
        sheetname=None,
        zxindustry_dummies=0,
        swindustry_dummies=0,
        only_cap=0,
        iplot=1,
        ilegend=1,
    ):
        """运行回测部分"""
        if comments_writer and not (comments_sheetname or sheetname):
            raise IOError("把total_comments输出到excel中时，必须指定sheetname🤒")
        if net_values_writer and not (net_values_sheetname or sheetname):
            raise IOError("把group_net_values输出到excel中时，必须指定sheetname🤒")
        if rets_writer and not (rets_sheetname or sheetname):
            raise IOError("把group_rets输出到excel中时，必须指定sheetname🤒")
        if neutralize:
            self.get_log_cap()
            self.get_neutral_factors(
                swindustry_dummies=swindustry_dummies,
                zxindustry_dummies=zxindustry_dummies,
            )
            self.deal_with_factors_after_neutralize()
        elif boxcox:
            self.get_log_cap(boxcox=True)
            self.get_neutral_factors(
                swindustry_dummies=swindustry_dummies,
                zxindustry_dummies=zxindustry_dummies,
                only_cap=only_cap,
            )
            self.deal_with_factors_after_neutralize()
        else:
            self.deal_with_factors()
        self.get_limit_ups_downs()
        self.get_data(groups_num)
        self.select_data_time(time_start, time_end)
        self.get_group_rets_net_values(
            groups_num=groups_num, value_weighted=value_weighted
        )
        self.get_long_short_comments(on_paper=on_paper)
        self.get_total_comments()
        if on_paper:
            group1_ttest = ss.ttest_1samp(self.group_rets.group1, 0).pvalue
            group10_ttest = ss.ttest_1samp(
                self.group_rets[f"group{groups_num}"], 0
            ).pvalue
            group_long_short_ttest = ss.ttest_1samp(self.long_short_rets, 0).pvalue
            group1_ret = self.group_rets.group1.mean()
            group10_ret = self.group_rets[f"group{groups_num}"].mean()
            group_long_short_ret = self.long_short_rets.mean()
            papers = pd.DataFrame(
                {
                    "评价指标": [
                        group1_ttest,
                        group10_ttest,
                        group_long_short_ttest,
                        group1_ret,
                        group10_ret,
                        group_long_short_ret,
                    ]
                },
                index=[
                    "分组1p值",
                    f"分组{groups_num}p值",
                    f"分组1-分组{groups_num}p值",
                    "分组1收益率",
                    f"分组{groups_num}收益率",
                    f"分组1-分组{groups_num}收益率",
                ],
            )
            self.total_comments = pd.concat([papers, self.total_comments])

        if plt_plot:
            if not STATES["NO_PLOT"]:
                if filename:
                    self.plot_net_values(
                        y2=y2, filename=filename, iplot=iplot, ilegend=bool(ilegend)
                    )
                else:
                    self.plot_net_values(
                        y2=y2,
                        filename=self.factors_file.split(".")[-2].split("/")[-1]
                        + str(groups_num)
                        + "分组",
                        iplot=iplot,
                        ilegend=bool(ilegend),
                    )
                plt.show()
        if plotly_plot:
            if not STATES["NO_PLOT"]:
                if filename:
                    self.plotly_net_values(filename=filename)
                else:
                    self.plotly_net_values(
                        filename=self.factors_file.split(".")[-2].split("/")[-1]
                        + str(groups_num)
                        + "分组"
                    )
        if print_comments:
            if not STATES["NO_COMMENT"]:
                tb = Texttable()
                tb.set_cols_width([8] * 4 + [12] + [8] * 2 + [7] * 2 + [8] + [10])
                tb.set_cols_dtype(["f"] * 11)
                tb.header(list(self.total_comments.T.columns))
                tb.add_rows(self.total_comments.T.to_numpy(), header=False)
                print(tb.draw())
        if sheetname:
            if comments_writer:
                if not on_paper:
                    total_comments = self.total_comments.copy()
                    tc = list(total_comments.评价指标)
                    tc[0] = str(round(tc[0] * 100, 2)) + "%"
                    tc[1] = str(round(tc[1], 2))
                    tc[2] = str(round(tc[2] * 100, 2)) + "%"
                    tc[3] = str(round(tc[3], 2))
                    tc[4] = str(round(tc[4], 2))
                    tc[5] = str(round(tc[5] * 100, 2)) + "%"
                    tc[6] = str(round(tc[6] * 100, 2)) + "%"
                    tc[7] = str(round(tc[7], 2))
                    tc[8] = str(round(tc[8] * 100, 2)) + "%"
                    tc[9] = str(round(tc[9] * 100, 2)) + "%"
                    tc[10] = str(round(tc[10] * 100, 2)) + "%"
                    new_total_comments = pd.DataFrame(
                        {sheetname: tc}, index=total_comments.index
                    )
                    new_total_comments.T.to_excel(comments_writer, sheet_name=sheetname)
                else:
                    self.total_comments.rename(columns={"评价指标": sheetname}).to_excel(
                        comments_writer, sheet_name=sheetname
                    )
            if net_values_writer:
                groups_net_values = self.group_net_values.copy()
                groups_net_values.index = groups_net_values.index.strftime("%Y/%m/%d")
                groups_net_values.columns = [
                    f"分组{i}" for i in range(1, len(list(groups_net_values.columns)))
                ] + ["多空对冲（右轴）"]
                groups_net_values.to_excel(net_values_writer, sheet_name=sheetname)
            if rets_writer:
                group_rets = self.group_rets.copy()
                group_rets.index = group_rets.index.strftime("%Y/%m/%d")
                group_rets.columns = [
                    f"分组{i}" for i in range(1, len(list(group_rets.columns)))
                ] + ["多空对冲（右轴）"]
                group_rets.to_excel(rets_writer, sheet_name=sheetname)
        else:
            if comments_writer and comments_sheetname:
                total_comments = self.total_comments.copy()
                tc = list(total_comments.评价指标)
                tc[0] = str(round(tc[0] * 100, 2)) + "%"
                tc[1] = str(round(tc[1], 2))
                tc[2] = str(round(tc[2] * 100, 2)) + "%"
                tc[3] = str(round(tc[3], 2))
                tc[4] = str(round(tc[4], 2))
                tc[5] = str(round(tc[5] * 100, 2)) + "%"
                tc[6] = str(round(tc[6] * 100, 2)) + "%"
                tc[7] = str(round(tc[7], 2))
                tc[8] = str(round(tc[8] * 100, 2)) + "%"
                tc[9] = str(round(tc[9] * 100, 2)) + "%"
                tc[10] = str(round(tc[10] * 100, 2)) + "%"
                new_total_comments = pd.DataFrame(
                    {comments_sheetname: tc}, index=total_comments.index
                )
                new_total_comments.T.to_excel(
                    comments_writer, sheet_name=comments_sheetname
                )
            if net_values_writer and net_values_sheetname:
                groups_net_values = self.group_net_values.copy()
                groups_net_values.index = groups_net_values.index.strftime("%Y/%m/%d")
                groups_net_values.columns = [
                    f"分组{i}" for i in range(1, len(list(groups_net_values.columns)))
                ] + ["多空对冲（右轴）"]
                groups_net_values.to_excel(
                    net_values_writer, sheet_name=net_values_sheetname
                )
            if rets_writer and rets_sheetname:
                group_rets = self.group_rets.copy()
                group_rets.index = group_rets.index.strftime("%Y/%m/%d")
                group_rets.columns = [
                    f"分组{i}" for i in range(1, len(list(group_rets.columns)))
                ] + ["多空对冲（右轴）"]
                group_rets.to_excel(rets_writer, sheet_name=rets_sheetname)


class pure_moonnight(object):
    """封装选股框架"""

    __slots__ = ["shen"]

    def __init__(
        self,
        factors: pd.DataFrame,
        groups_num: int = 10,
        neutralize: bool = 0,
        boxcox: bool = 1,
        value_weighted: bool = 0,
        y2: bool = 0,
        plt_plot: bool = 1,
        plotly_plot: bool = 0,
        filename: str = "分组净值图",
        time_start: int = None,
        time_end: int = None,
        print_comments: bool = 1,
        comments_writer: pd.ExcelWriter = None,
        net_values_writer: pd.ExcelWriter = None,
        rets_writer: pd.ExcelWriter = None,
        comments_sheetname: str = None,
        net_values_sheetname: str = None,
        rets_sheetname: str = None,
        on_paper: bool = 0,
        sheetname: str = None,
        zxindustry_dummies: bool = 0,
        swindustry_dummies: bool = 0,
        ages: pd.DataFrame = None,
        sts: pd.DataFrame = None,
        states: pd.DataFrame = None,
        opens: pd.DataFrame = None,
        closes: pd.DataFrame = None,
        capitals: pd.DataFrame = None,
        swindustry_dummy: pd.DataFrame = None,
        zxindustry_dummy: pd.DataFrame = None,
        no_read_indu: bool = 0,
        only_cap: bool = 0,
        iplot: bool = 1,
        ilegend: bool = 1,
    ) -> None:
        """一键回测框架，测试单因子的月频调仓的分组表现
        每月月底计算因子值，月初第一天开盘时买入，月末收盘最后一天收盘时卖出
        剔除上市不足60天的，停牌天数超过一半的，st天数超过一半的
        月末收盘跌停的不卖出，月初开盘涨停的不买入
        由最好组和最差组的多空组合构成多空对冲组

        Parameters
        ----------
        factors : pd.DataFrame
            要用于检测的因子值，index是时间，columns是股票代码
        groups_num : int, optional
            分组数量, by default 10
        neutralize : bool, optional
            对流通市值取自然对数，以完成行业市值中性化, by default 0
        boxcox : bool, optional
            对流通市值做截面boxcox变换，以完成行业市值中性化, by default 1
        value_weighted : bool, optional
            是否用流通市值加权, by default 0
        y2 : bool, optional
            画图时是否启用第二y轴, by default 0
        plt_plot : bool, optional
            将分组净值曲线用matplotlib画出来, by default 1
        plotly_plot : bool, optional
            将分组净值曲线用plotly画出来, by default 0
        filename : str, optional
            分组净值曲线的图保存的名称, by default "分组净值图"
        time_start : int, optional
            回测起始时间（此参数已废弃，请在因子上直接截断）, by default None
        time_end : int, optional
            回测终止时间（此参数已废弃，请在因子上直接截断）, by default None
        print_comments : bool, optional
            是否打印出评价指标, by default 1
        comments_writer : pd.ExcelWriter, optional
            用于记录评价指标的xlsx文件, by default None
        net_values_writer : pd.ExcelWriter, optional
            用于记录净值序列的xlsx文件, by default None
        rets_writer : pd.ExcelWriter, optional
            用于记录收益率序列的xlsx文件, by default None
        comments_sheetname : str, optional
            在记录评价指标的xlsx文件中，该工作表的名称, by default None
        net_values_sheetname : str, optional
            在记录净值序列的xlsx文件中，该工作表的名称, by default None
        rets_sheetname : str, optional
            在记录收益率序列的xlsx文件中，该工作表的名称, by default None
        on_paper : bool, optional
            使用学术化评价指标, by default 0
        sheetname : str, optional
            各个pd.Excelwriter中工作表的统一名称, by default None
        zxindustry_dummies : bool, optional
            行业中性化时，选用中信一级行业, by default 0
        swindustry_dummies : bool, optional
            行业中性化时，选用申万一级行业, by default 0
        ages : pd.DataFrame, optional
            输入股票上市天数的数据，index是时间，columns是股票代码，values是天数, by default None
        sts : pd.DataFrame, optional
            输入股票每天是否st的数据，是st股即为1，否则为0，index是时间，columns是股票代码，values是0或1, by default None
        states : pd.DataFrame, optional
            输入股票每天交易状态的数据，正常交易为1，否则为0，index是时间，columns是股票代码，values是0或1, by default None
        opens : pd.DataFrame, optional
            输入股票的复权开盘价数据，index是时间，columns是股票代码，values是价格, by default None
        closes : pd.DataFrame, optional
            输入股票的复权收盘价数据，index是时间，columns是股票代码，values是价格, by default None
        capitals : pd.DataFrame, optional
            输入股票的每月月末流通市值数据，index是时间，columns是股票代码，values是流通市值, by default None
        swindustry_dummy : pd.DataFrame, optioanl
            熟人股票的每月月末的申万一级行业哑变量，表包含33列，第一列为股票代码，名为`code`，第二列为月末最后一天的日期，名为`date`
            其余31列，为各个行业的哑变量，名为`w1`、`w2`、`w3`……`w31`, by default None
        zxindustry_dummy : pd.DataFrame, optioanl
            熟人股票的每月月末的中信一级行业哑变量，表包含32列，第一列为股票代码，名为`code`，第二列为月末最后一天的日期，名为`date`
            其余30列，为各个行业的哑变量，名为`w1`、`w2`、`w3`……`w30`, by default None
        no_read_indu : bool, optional
            不读入行业数据, by default 0
        only_cap : bool, optional
            仅做市值中性化, by default 0
        iplot : bool, optional
            使用cufflinks呈现回测结果, by default 1
        ilegend : bool, optional
            使用cufflinks绘图时，是否显示图例, by default 1
        """

        if not isinstance(factors, pd.DataFrame):
            factors = factors()
        start = datetime.datetime.strftime(factors.index.min(), "%Y%m%d")
        if ages is None:
            ages = read_daily(age=1, start=start)
        if sts is None:
            sts = read_daily(st=1, start=start)
        if states is None:
            states = read_daily(state=1, start=start)
        if opens is None:
            opens = read_daily(open=1, start=start)
        if closes is None:
            closes = read_daily(close=1, start=start)
        if capitals is None:
            capitals = read_daily(flow_cap=1, start=start).resample("M").last()
        if comments_writer is None and sheetname is not None:
            from pure_ocean_breeze.legacy_version.v3p4.state.states import COMMENTS_WRITER

            comments_writer = COMMENTS_WRITER
        if net_values_writer is None and sheetname is not None:
            from pure_ocean_breeze.legacy_version.v3p4.state.states import NET_VALUES_WRITER

            net_values_writer = NET_VALUES_WRITER
        if not on_paper:
            from pure_ocean_breeze.legacy_version.v3p4.state.states import ON_PAPER

            on_paper = ON_PAPER
        from pure_ocean_breeze.legacy_version.v3p4.state.states import MOON_START

        if MOON_START is not None:
            factors = factors[factors.index >= pd.Timestamp(str(MOON_START))]
        from pure_ocean_breeze.legacy_version.v3p4.state.states import MOON_END

        if MOON_END is not None:
            factors = factors[factors.index <= pd.Timestamp(str(MOON_END))]
        if boxcox + neutralize == 0:
            no_read_indu = 1
        if only_cap + no_read_indu > 0:
            only_cap = no_read_indu = 1
        if iplot:
            print_comments = 0
        self.shen = pure_moon(
            no_read_indu=no_read_indu,
            swindustry_dummy=swindustry_dummy,
            zxindustry_dummy=zxindustry_dummy,
            read_in_swindustry_dummy=swindustry_dummies,
        )
        self.shen.set_basic_data(
            age=ages,
            st=sts,
            state=states,
            open=opens,
            close=closes,
            capital=capitals,
        )
        self.shen.set_factor_df_date_as_index(factors)
        self.shen.prerpare()
        self.shen.run(
            groups_num=groups_num,
            neutralize=neutralize,
            boxcox=boxcox,
            value_weighted=value_weighted,
            y2=y2,
            plt_plot=plt_plot,
            plotly_plot=plotly_plot,
            filename=filename,
            time_start=time_start,
            time_end=time_end,
            print_comments=print_comments,
            comments_writer=comments_writer,
            net_values_writer=net_values_writer,
            rets_writer=rets_writer,
            comments_sheetname=comments_sheetname,
            net_values_sheetname=net_values_sheetname,
            rets_sheetname=rets_sheetname,
            on_paper=on_paper,
            sheetname=sheetname,
            swindustry_dummies=swindustry_dummies,
            zxindustry_dummies=zxindustry_dummies,
            only_cap=only_cap,
            iplot=iplot,
            ilegend=ilegend,
        )

    def __call__(self) -> pd.DataFrame:
        """如果做了行业市值中性化，则返回行业市值中性化之后的因子数据

        Returns
        -------
        `pd.DataFrame`
            如果做了行业市值中性化，则行业市值中性化之后的因子数据，否则返回原因子数据
        """
        return self.shen.factors_out

    def comments_ten(self) -> pd.DataFrame:
        """对回测的十分组结果分别给出评价

        Returns
        -------
        `pd.DataFrame`
            评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率
        """
        rets_cols = list(self.shen.group_rets.columns)
        rets_cols = rets_cols[:-1]
        coms = []
        for i in rets_cols:
            ret = self.shen.group_rets[i]
            net = self.shen.group_net_values[i]
            com = comments_on_twins(net, ret)
            com = com.to_frame(i)
            coms.append(com)
        df = pd.concat(coms, axis=1)
        return df.T


class pure_fall(object):
    # DONE：修改为因子文件名可以带“日频_“，也可以不带“日频_“
    def __init__(self, daily_path: str) -> None:
        """一个使用mysql中的分钟数据，来更新因子值的框架

        Parameters
        ----------
        daily_path : str
            日频因子值存储文件的名字，请以'.feather'结尾
        """
        self.homeplace = HomePlace()
        # 将分钟数据拼成一张日频因子表
        self.daily_factors = None
        self.daily_factors_path = self.homeplace.factor_data_file + "日频_" + daily_path

    def __call__(self, monthly=False):
        """为了防止属性名太多，忘记了要调用哪个才是结果，因此可以直接输出月度数据表"""
        if monthly:
            return self.monthly_factors.copy()
        else:
            try:
                return self.daily_factors.copy()
            except Exception:
                return self.monthly_factors.copy()

    def __add__(self, selfas):
        """将几个因子截面标准化之后，因子值相加"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2s = []
        if not isinstance(selfas, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfas} is changed into Iterable")
            selfas = (selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 + i
        new_pure = pure_fall()
        new_pure.monthly_factors = fac1
        return new_pure

    def __mul__(self, selfas):
        """将几个因子横截面标准化之后，使其都为正数，然后因子值相乘"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac1 = fac1 - fac1.min()
        fac2s = []
        if not isinstance(selfas, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfas} is changed into Iterable")
            selfas = (selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2 = fac2 - fac2.min()
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 * i
        new_pure = pure_fall()
        new_pure.monthly_factors = fac1
        return new_pure

    def __truediv__(self, selfa):
        """两个一正一副的因子，可以用此方法相减"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
        fac = fac1 - fac2
        new_pure = pure_fall()
        new_pure.monthly_factors = fac
        return new_pure

    def __floordiv__(self, selfa):
        """两个因子一正一负，可以用此方法相除"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
        fac1 = fac1 - fac1.min()
        fac2 = fac2 - fac2.min()
        fac = fac1 / fac2
        fac = fac.replace(np.inf, np.nan)
        new_pure = pure_fall()
        new_pure.monthly_factors = fac
        return new_pure

    @kk.desktop_sender(title="嘿，正交化结束啦～🐬")
    def __sub__(self, selfa):
        """用主因子剔除其他相关因子、传统因子等
        selfa可以为多个因子对象组成的元组或列表，每个辅助因子只需要有月度因子文件路径即可"""
        if is_notebook():
            tqdm.tqdm_notebook().pandas()
        else:
            tqdm.tqdm.pandas()
        if not isinstance(selfa, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfa} is changed into Iterable")
            selfa = (selfa,)
        fac_main = self.wide_to_long(self.monthly_factors, "fac")
        fac_helps = [i.monthly_factors for i in selfa]
        help_names = ["help" + str(i) for i in range(1, (len(fac_helps) + 1))]
        fac_helps = list(map(self.wide_to_long, fac_helps, help_names))
        fac_helps = pd.concat(fac_helps, axis=1)
        facs = pd.concat([fac_main, fac_helps], axis=1).dropna()
        facs = facs.groupby("date").progress_apply(
            lambda x: self.de_in_group(x, help_names)
        )
        facs = facs.unstack()
        facs.columns = list(map(lambda x: x[1], list(facs.columns)))
        return facs

    def __gt__(self, selfa):
        """用于输出25分组表格，使用时，以x>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子"""
        x = self.monthly_factors.copy()
        y = selfa.monthly_factors.copy()
        x = x.stack().reset_index()
        y = y.stack().reset_index()
        x.columns = ["date", "code", "fac"]
        y.columns = ["date", "code", "fac"]
        shen = pure_moon()
        x = x.groupby("date").apply(lambda df: shen.get_groups(df, 5))
        x = (
            x.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupx"})
        )
        xy = pd.merge(x, y, on=["date", "code"])
        xy = xy.groupby(["date", "groupx"]).apply(lambda df: shen.get_groups(df, 5))
        xy = (
            xy.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupy"})
        )
        xy = xy.assign(fac=xy.groupx * 5 + xy.groupy)
        xy = xy[["date", "code", "fac"]]
        xy = xy.set_index(["date", "code"]).unstack()
        xy.columns = [i[1] for i in list(xy.columns)]
        new_pure = pure_fall()
        new_pure.monthly_factors = xy
        return new_pure

    def __rshift__(self, selfa):
        """用于输出100分组表格，使用时，以x>>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子"""
        x = self.monthly_factors.copy()
        y = selfa.monthly_factors.copy()
        x = x.stack().reset_index()
        y = y.stack().reset_index()
        x.columns = ["date", "code", "fac"]
        y.columns = ["date", "code", "fac"]
        shen = pure_moon()
        x = x.groupby("date").apply(lambda df: shen.get_groups(df, 10))
        x = (
            x.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupx"})
        )
        xy = pd.merge(x, y, on=["date", "code"])
        xy = xy.groupby(["date", "groupx"]).apply(lambda df: shen.get_groups(df, 10))
        xy = (
            xy.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupy"})
        )
        xy = xy.assign(fac=xy.groupx * 10 + xy.groupy)
        xy = xy[["date", "code", "fac"]]
        xy = xy.set_index(["date", "code"]).unstack()
        xy.columns = [i[1] for i in list(xy.columns)]
        new_pure = pure_fall()
        new_pure.monthly_factors = xy
        return new_pure

    def wide_to_long(self, df, i):
        """将宽数据转化为长数据，用于因子表转化和拼接"""
        df = df.stack().reset_index()
        df.columns = ["date", "code", i]
        df = df.set_index(["date", "code"])
        return df

    def de_in_group(self, df, help_names):
        """对每个时间，分别做回归，剔除相关因子"""
        ols_order = "fac~" + "+".join(help_names)
        ols_result = smf.ols(ols_order, data=df).fit()
        params = {i: ols_result.params[i] for i in help_names}
        predict = [params[i] * df[i] for i in help_names]
        predict = reduce(lambda x, y: x + y, predict)
        df.fac = df.fac - predict - ols_result.params["Intercept"]
        df = df[["fac"]]
        return df

    def standardlize_in_cross_section(self, df):
        """
        在横截面上做标准化
        输入的df应为，列名是股票代码，索引是时间
        """
        df = df.T
        df = (df - df.mean()) / df.std()
        df = df.T
        return df

    def get_single_day_factor(self, func: Callable, day: int) -> pd.DataFrame:
        """计算单日的因子值，通过sql数据库，读取单日的数据，然后计算因子值"""
        sql = sqlConfig("minute_data_stock_alter")
        df = sql.get_data(str(day))
        the_func = partial(func)
        df = df.groupby(["code"]).apply(the_func).to_frame()
        df.columns = [str(day)]
        df = df.T
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        return df

    @kk.desktop_sender(title="嘿，分钟数据处理完啦～🎈")
    def get_daily_factors_alter(self, func: Callable) -> None:
        """用mysql逐日更新分钟数据构造的因子

        Parameters
        ----------
        func : Callable
            构造分钟数据使用的函数

        Raises
        ------
        `IOError`
            如果没有历史因子数据，将报错
        """
        """通过minute_data_stock_alter数据库一天一天计算因子值"""
        try:
            try:
                self.daily_factors = pd.read_feather(self.daily_factors_path)
            except Exception:
                self.daily_factors_path = self.daily_factors_path.split("日频_")
                self.daily_factors_path = (
                    self.daily_factors_path[0] + self.daily_factors_path[1]
                )
                self.daily_factors = pd.read_feather(self.daily_factors_path)
            self.daily_factors = self.daily_factors.rename(
                columns={list(self.daily_factors.columns)[0]: "date"}
            )
            self.daily_factors = self.daily_factors.drop_duplicates(
                subset=["date"], keep="last"
            )
            self.daily_factors = self.daily_factors.set_index("date")
            sql = sqlConfig("minute_data_stock_alter")
            now_minute_datas = sql.show_tables(full=False)
            now_minute_data = now_minute_datas[-1]
            now_minute_data = pd.Timestamp(now_minute_data)
            if self.daily_factors.index.max() < now_minute_data:
                if not STATES["NO_LOG"]:
                    logger.info(
                        f"上次存储的因子值到{self.daily_factors.index.max()}，而分钟数据最新到{now_minute_data}，开始更新……"
                    )
                old_end = datetime.datetime.strftime(
                    self.daily_factors.index.max(), "%Y%m%d"
                )
                now_minute_datas = [i for i in now_minute_datas if i > old_end]
                dfs = []
                for c in tqdm.tqdm(now_minute_datas, desc="桂棹兮兰桨，击空明兮遂流光🌊"):
                    df = self.get_single_day_factor(func, c)
                    dfs.append(df)
                dfs = pd.concat(dfs)
                dfs = dfs.sort_index()
                self.daily_factors = pd.concat([self.daily_factors, dfs])
                self.daily_factors = self.daily_factors.dropna(how="all")
                self.daily_factors = self.daily_factors[
                    self.daily_factors.index >= pd.Timestamp(str(STATES["START"]))
                ]
            self.daily_factors = self.daily_factors.reset_index()
            self.daily_factors = self.daily_factors.rename(
                columns={list(self.daily_factors.columns)[0]: "date"}
            )
            self.daily_factors = self.daily_factors.drop_duplicates(
                subset=["date"], keep="first"
            )
            self.daily_factors = self.daily_factors.set_index("date")
            self.daily_factors.reset_index().to_feather(self.daily_factors_path)
            if not STATES["NO_LOG"]:
                logger.success("更新已完成")

        except Exception:
            raise IOError(
                "您还没有该因子的初级数据，暂时不能更新。请先使用pure_fall_frequent或pure_fall_flexible计算历史因子值。"
            )


class pure_fallmount(pure_fall):
    """继承自父类，专为做因子截面标准化之后相加和因子剔除其他辅助因子的作用"""

    def __init__(self, monthly_factors):
        """输入月度因子值，以设定新的对象"""
        super(pure_fall, self).__init__()
        self.monthly_factors = monthly_factors

    def __call__(self, monthly=False):
        """为了防止属性名太多，忘记了要调用哪个才是结果，因此可以直接输出月度数据表"""
        if monthly:
            return self.monthly_factors.copy()
        else:
            try:
                return self.daily_factors.copy()
            except Exception:
                return self.monthly_factors.copy()

    def __add__(self, selfas):
        """返回一个对象，而非一个表格，如需表格请调用对象"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2s = []
        if not isinstance(selfas, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfas} is changed into Iterable")
            selfas = (selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 + i
        new_pure = pure_fallmount(fac1)
        return new_pure

    def __mul__(self, selfas):
        """将几个因子横截面标准化之后，使其都为正数，然后因子值相乘"""
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac1 = fac1 - fac1.min()
        fac2s = []
        if not isinstance(selfas, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfas} is changed into Iterable")
            selfas = (selfas,)
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2 = fac2 - fac2.min()
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 * i
        new_pure = pure_fall()
        new_pure.monthly_factors = fac1
        return new_pure

    def __sub__(self, selfa):
        """返回对象，如需表格，请调用对象"""
        if is_notebook():
            tqdm.tqdm_notebook().pandas()
        else:
            tqdm.tqdm.pandas()
        if not isinstance(selfa, Iterable):
            if not STATES["NO_LOG"]:
                logger.warning(f"{selfa} is changed into Iterable")
            selfa = (selfa,)
        fac_main = self.wide_to_long(self.monthly_factors, "fac")
        fac_helps = [i.monthly_factors for i in selfa]
        help_names = ["help" + str(i) for i in range(1, (len(fac_helps) + 1))]
        fac_helps = list(map(self.wide_to_long, fac_helps, help_names))
        fac_helps = pd.concat(fac_helps, axis=1)
        facs = pd.concat([fac_main, fac_helps], axis=1).dropna()
        facs = facs.groupby("date").progress_apply(
            lambda x: self.de_in_group(x, help_names)
        )
        facs = facs.unstack()
        facs.columns = list(map(lambda x: x[1], list(facs.columns)))
        new_pure = pure_fallmount(facs)
        return new_pure

    def __gt__(self, selfa):
        """用于输出25分组表格，使用时，以x>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子"""
        x = self.monthly_factors.copy()
        y = selfa.monthly_factors.copy()
        x = x.stack().reset_index()
        y = y.stack().reset_index()
        x.columns = ["date", "code", "fac"]
        y.columns = ["date", "code", "fac"]
        shen = pure_moon()
        x = x.groupby("date").apply(lambda df: shen.get_groups(df, 5))
        x = (
            x.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupx"})
        )
        xy = pd.merge(x, y, on=["date", "code"])
        xy = xy.groupby(["date", "groupx"]).apply(lambda df: shen.get_groups(df, 5))
        xy = (
            xy.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupy"})
        )
        xy = xy.assign(fac=xy.groupx * 5 + xy.groupy)
        xy = xy[["date", "code", "fac"]]
        xy = xy.set_index(["date", "code"]).unstack()
        xy.columns = [i[1] for i in list(xy.columns)]
        new_pure = pure_fallmount(xy)
        return new_pure

    def __rshift__(self, selfa):
        """用于输出100分组表格，使用时，以x>>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子"""
        x = self.monthly_factors.copy()
        y = selfa.monthly_factors.copy()
        x = x.stack().reset_index()
        y = y.stack().reset_index()
        x.columns = ["date", "code", "fac"]
        y.columns = ["date", "code", "fac"]
        shen = pure_moon()
        x = x.groupby("date").apply(lambda df: shen.get_groups(df, 10))
        x = (
            x.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupx"})
        )
        xy = pd.merge(x, y, on=["date", "code"])
        xy = xy.groupby(["date", "groupx"]).apply(lambda df: shen.get_groups(df, 10))
        xy = (
            xy.reset_index(drop=True)
            .drop(columns=["fac"])
            .rename(columns={"group": "groupy"})
        )
        xy = xy.assign(fac=xy.groupx * 10 + xy.groupy)
        xy = xy[["date", "code", "fac"]]
        xy = xy.set_index(["date", "code"]).unstack()
        xy.columns = [i[1] for i in list(xy.columns)]
        new_pure = pure_fallmount(xy)
        return new_pure


class pure_fall_frequent(object):
    """对单只股票单日进行操作"""

    def __init__(
        self,
        factor_file: str,
        startdate: int = None,
        enddate: int = None,
        kind: str = "stock",
        clickhouse: bool = 0,
        questdb: bool = 0,
        ignore_history_in_questdb: bool = 0,
        groupby_target: list = ["date", "code"],
    ) -> None:
        """基于clickhouse的分钟数据，计算因子值，每天的因子值只用到当日的数据

        Parameters
        ----------
        factor_file : str
            用于保存因子值的文件名，需为feather文件，以'.feather'结尾
        startdate : int, optional
            起始时间，形如20121231，为开区间, by default None
        enddate : int, optional
            截止时间，形如20220814，为闭区间，为空则计算到最近数据, by default None
        kind : str, optional
            类型为股票还是指数，指数为'index', by default "stock"
        clickhouse : bool, optional
            使用clickhouse作为数据源，如果postgresql与本参数都为0，将依然从clickhouse中读取, by default 0
        questdb : bool, optional
            使用questdb作为数据源, by default 0
        ignore_history_in_questdb : bool, optional
            打断后重新从头计算，清除在questdb中的记录
        groupby_target: list, optional
            groupby计算时，分组的依据，使用此参数时，自定义函数的部分，如果指定按照['date']分组groupby计算，
            则返回时，应当返回一个两列的dataframe，第一列为股票代码，第二列为为因子值, by default ['date','code']
        """
        homeplace = HomePlace()
        self.kind = kind
        self.groupby_target = groupby_target
        if clickhouse == 0 and questdb == 0:
            clickhouse = 1
        self.clickhouse = clickhouse
        self.questdb = questdb
        if clickhouse == 1:
            # 连接clickhouse
            self.chc = ClickHouseClient("minute_data")
        elif questdb == 1:
            self.chc = Questdb()
        # 将计算到一半的因子，存入questdb中，避免中途被打断后重新计算，表名即为因子文件名的汉语拼音
        pinyin = Pinyin()
        self.factor_file_pinyin = pinyin.get_pinyin(
            factor_file.replace(".feather", ""), ""
        )
        self.factor_steps = Questdb()
        # 完整的因子文件路径
        factor_file = homeplace.factor_data_file + factor_file
        self.factor_file = factor_file
        # 读入之前的因子
        if os.path.exists(factor_file):
            factor_old = pd.read_feather(self.factor_file)
            factor_old.columns = ["date"] + list(factor_old.columns)[1:]
            factor_old = factor_old.drop_duplicates(subset=["date"])
            factor_old = factor_old.set_index("date")
            self.factor_old = factor_old
            # 已经算好的日子
            dates_old = sorted(list(factor_old.index.strftime("%Y%m%d").astype(int)))
            self.dates_old = dates_old
        elif (not ignore_history_in_questdb) and self.factor_file_pinyin in list(
            self.factor_steps.get_data("show tables").table
        ):
            logger.info(
                f"上次计算途中被打断，已经将数据备份在questdb数据库的表{self.factor_file_pinyin}中，现在将读取上次的数据，继续计算"
            )
            factor_old = self.factor_steps.get_data(
                f"select * from {self.factor_file_pinyin}"
            )
            # 判断一下每天是否生成多个数据，单个数据就以float形式存储，多个数据以list形式存储
            if "f0" in list(factor_old.columns):
                factor_old = factor_old[factor_old.f0 != "date"]
                factor_old.columns = ["date", "code", "fac"]
                try:
                    factor_old.fac = factor_old.fac.apply(
                        lambda x: [float(i) for i in x[1:-1].split(" ") if i != ""]
                    )
                except Exception:
                    factor_old.fac = factor_old.fac.apply(
                        lambda x: [float(i) for i in x[1:-1].split(", ") if i != ""]
                    )
            factor_old = factor_old.pivot(index="date", columns="code", values="fac")
            factor_old.index = pd.to_datetime(factor_old.index)
            factor_old = factor_old.sort_index()
            factor_old = drop_duplicates_index(factor_old)
            self.factor_old = factor_old
            # 已经算好的日子
            dates_old = sorted(list(factor_old.index.strftime("%Y%m%d").astype(int)))
            self.dates_old = dates_old
        elif ignore_history_in_questdb and self.factor_file_pinyin in list(
            self.factor_steps.get_data("show tables").table
        ):
            logger.info(
                f"上次计算途中被打断，已经将数据备份在questdb数据库的表{self.factor_file_pinyin}中，但您选择重新计算，所以正在删除原来的数据，从头计算"
            )
            factor_old = self.factor_steps.do_order(
                f"drop table {self.factor_file_pinyin}"
            )
            self.factor_old = None
            self.dates_old = []
            logger.info("删除完毕，正在重新计算")
        else:
            self.factor_old = None
            self.dates_old = []
            logger.info("这个因子以前没有，正在重新计算")
        # 读取当前所有的日子
        dates_all = self.chc.show_all_dates(f"minute_data_{kind}")
        dates_all = [int(i) for i in dates_all]
        if startdate is None:
            ...
        else:
            dates_all = [i for i in dates_all if i >= startdate]
        if enddate is None:
            ...
        else:
            dates_all = [i for i in dates_all if i <= enddate]
        self.dates_all = dates_all
        # 需要新补充的日子
        self.dates_new = sorted([i for i in dates_all if i not in self.dates_old])
        if len(self.dates_new) == 0:
            ...
        elif len(self.dates_new) == 1:
            self.dates_new_intervals = [[pd.Timestamp(str(self.dates_new[0]))]]
            print(f"只缺一天{self.dates_new[0]}")
        else:
            dates = [pd.Timestamp(str(i)) for i in self.dates_new]
            intervals = [[]] * len(dates)
            interbee = 0
            intervals[0] = intervals[0] + [dates[0]]
            for i in range(len(dates) - 1):
                val1 = dates[i]
                val2 = dates[i + 1]
                if val2 - val1 < pd.Timedelta(days=30):
                    ...
                else:
                    interbee = interbee + 1
                intervals[interbee] = intervals[interbee] + [val2]
            intervals = [i for i in intervals if len(i) > 0]
            print(f"共{len(intervals)}个时间区间，分别是")
            for date in intervals:
                print(f"从{date[0]}到{date[-1]}")
            self.dates_new_intervals = intervals
        self.factor_new = []

    def __call__(self) -> pd.DataFrame:
        """获得经运算产生的因子

        Returns
        -------
        `pd.DataFrame`
            经运算产生的因子值
        """
        return self.factor.copy()

    def select_one_calculate(
        self,
        date: pd.Timestamp,
        func: Callable,
        fields: str = "*",
        show_time: bool = 0,
    ) -> None:
        the_func = partial(func)
        if not isinstance(date, int):
            date = int(datetime.datetime.strftime(date, "%Y%m%d"))
        # 开始计算因子值
        if self.clickhouse == 1:
            sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={date * 100} order by code,date,num"
        else:
            sql_order = (
                f"select {fields} from minute_data_{self.kind} where date='{date}'"
            )
        if show_time:
            df = self.chc.get_data_show_time(sql_order)
        else:
            df = self.chc.get_data(sql_order)
        if self.clickhouse == 1:
            df = ((df.set_index("code")) / 100).reset_index()
        else:
            df.num = df.num.astype(int)
            df.date = df.date.astype(int)
            df = df.sort_values(["date", "num"])
        df = df.groupby(self.groupby_target).apply(the_func)
        if self.groupby_target == ["date", "code"]:
            df = df.to_frame("fac").reset_index()
            df.columns = ["date", "code", "fac"]
        else:
            df = df.reset_index()
        df = df.pivot(columns="code", index="date", values="fac")
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
        return df

    def select_many_calculate(
        self,
        dates: list[pd.Timestamp],
        func: Callable,
        fields: str = "*",
        chunksize: int = 10,
        show_time: bool = 0,
        tqdm_inside: bool = 0,
    ) -> None:
        the_func = partial(func)
        dates = [int(datetime.datetime.strftime(i, "%Y%m%d")) for i in dates]
        # 将需要更新的日子分块，每200天一组，一起运算
        dates_new_len = len(dates)
        cut_points = list(range(0, dates_new_len, chunksize)) + [dates_new_len - 1]
        if cut_points[-1] == cut_points[-2]:
            cut_points = cut_points[:-1]
        cut_first = cut_points[0]
        cuts = tuple(zip(cut_points[:-1], cut_points[1:]))
        print(f"共{len(cuts)}段")
        factor_new = []
        df_first = self.select_one_calculate(
            date=dates[0],
            func=func,
            fields=fields,
            show_time=show_time,
        )
        factor_new.append(df_first)
        to_save = df_first.stack().reset_index()
        to_save.columns = ["date", "code", "fac"]
        self.factor_steps.write_via_csv(to_save, self.factor_file_pinyin)

        if tqdm_inside == 1:
            # 开始计算因子值
            for date1, date2 in cuts:
                if self.clickhouse == 1:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{dates[date1] * 100} and date<={dates[date2] * 100} order by code,date,num"
                else:
                    sql_order = f"select {fields} from minute_data_{self.kind} where cast(date as int)>{dates[date1]} and cast(date as int)<={dates[date2]}"
                if show_time:
                    df = self.chc.get_data_show_time(sql_order)
                else:
                    df = self.chc.get_data(sql_order)
                if self.clickhouse == 1:
                    df = ((df.set_index("code")) / 100).reset_index()
                else:
                    df.num = df.num.astype(int)
                    df.date = df.date.astype(int)
                    df = df.sort_values(["date", "num"])
                if is_notebook():
                    tqdm.tqdm_notebook().pandas()
                else:
                    tqdm.tqdm.pandas()
                df = df.groupby(self.groupby_target).progress_apply(the_func)
                df = df.to_frame("fac").reset_index()
                df.columns = ["date", "code", "fac"]
                df = df.pivot(columns="code", index="date", values="fac")
                df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
                factor_new.append(df)
                to_save = df.stack().reset_index()
                to_save.columns = ["date", "code", "fac"]
                self.factor_steps.write_via_csv(to_save, self.factor_file_pinyin)
        else:
            if is_notebook():
                # 开始计算因子值
                for date1, date2 in tqdm.tqdm_notebook(cuts, desc="不知乘月几人归，落月摇情满江树。"):
                    if self.clickhouse == 1:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{dates[date1] * 100} and date<={dates[date2] * 100} order by code,date,num"
                    else:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{dates[date1]} and date<={dates[date2]} order by code,date,num"
                    if show_time:
                        df = self.chc.get_data_show_time(sql_order)
                    else:
                        df = self.chc.get_data(sql_order)
                    if self.clickhouse == 1:
                        df = ((df.set_index("code")) / 100).reset_index()
                    df = df.groupby(self.groupby_target).apply(the_func)
                    if self.groupby_target == ["date", "code"]:
                        df = df.to_frame("fac").reset_index()
                        df.columns = ["date", "code", "fac"]
                    else:
                        df = df.reset_index()
                    df = df.pivot(columns="code", index="date", values="fac")
                    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
                    factor_new.append(df)
                    to_save = df.stack().reset_index()
                    to_save.columns = ["date", "code", "fac"]
                    self.factor_steps.write_via_csv(to_save, self.factor_file_pinyin)
            else:
                # 开始计算因子值
                for date1, date2 in tqdm.tqdm(cuts, desc="不知乘月几人归，落月摇情满江树。"):
                    if self.clickhouse == 1:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{dates[date1] * 100} and date<={dates[date2] * 100} order by code,date,num"
                    else:
                        sql_order = f"select {fields} from minute_data_{self.kind} where cast(date as int)>{dates[date1]} and cast(date as int)<={dates[date2]}"
                    if show_time:
                        df = self.chc.get_data_show_time(sql_order)
                    else:
                        df = self.chc.get_data(sql_order)
                    if self.clickhouse == 1:
                        df = ((df.set_index("code")) / 100).reset_index()
                    else:
                        df.num = df.num.astype(int)
                        df.date = df.date.astype(int)
                        df = df.sort_values(["date", "num"])
                    df = df.groupby(self.groupby_target).apply(the_func)
                    if self.groupby_target == ["date", "code"]:
                        df = df.to_frame("fac").reset_index()
                        df.columns = ["date", "code", "fac"]
                    else:
                        df = df.reset_index()
                    df = df.pivot(columns="code", index="date", values="fac")
                    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
                    factor_new.append(df)
                    to_save = df.stack().reset_index()
                    to_save.columns = ["date", "code", "fac"]
                    self.factor_steps.write_via_csv(to_save, self.factor_file_pinyin)
        factor_new = pd.concat(factor_new)
        return factor_new

    def select_any_calculate(
        self,
        dates: list[pd.Timestamp],
        func: Callable,
        fields: str = "*",
        chunksize: int = 10,
        show_time: bool = 0,
        tqdm_inside: bool = 0,
    ) -> None:
        if len(dates) == 1:
            res = self.select_one_calculate(
                dates[0],
                func=func,
                fields=fields,
                show_time=show_time,
            )
        else:
            res = self.select_many_calculate(
                dates=dates,
                func=func,
                fields=fields,
                chunksize=chunksize,
                show_time=show_time,
                tqdm_inside=tqdm_inside,
            )
        return res

    @staticmethod
    def for_cross_via_str(func):
        """返回值为两层的list，每一个里层的小list为单个股票在这一天的返回值
        例如
        ```python
        return [[0.11,0.24,0.55],[2.59,1.99,0.43],[1.32,8.88,7.77]……]
        ```
        上例中，每个股票一天返回三个因子值，里层的list按照股票代码顺序排列"""

        def full_run(df, *args, **kwargs):
            codes = sorted(list(set(df.code)))
            res = func(df, *args, **kwargs)
            if isinstance(res[0], list):
                kind = 1
                res = [",".join(i) for i in res]
            else:
                kind = 0
            df = pd.DataFrame({"code": codes, "fac": res})
            if kind:
                df.fac = df.fac.apply(lambda x: [float(i) for i in x.split(",")])
            return df

        return full_run

    @staticmethod
    def for_cross_via_zip(func):
        """返回值为多个pd.Series，每个pd.Series的index为股票代码，values为单个因子值
        例如
        ```python
        return (
                    pd.Series([1.54,8.77,9.99……],index=['000001.SZ','000002.SZ','000004.SZ'……]),
                    pd.Series([3.54,6.98,9.01……],index=['000001.SZ','000002.SZ','000004.SZ'……]),
                )
        ```
        上例中，每个股票一天返回两个因子值，每个pd.Series对应一个因子值
        """

        def full_run(df, *args, **kwargs):
            res = func(df, *args, **kwargs)
            res = pd.concat(res, axis=1)
            res.columns = [f"fac{i}" for i in range(len(res.columns))]
            res = res.assign(fac=list(zip(*[res[i] for i in list(res.columns)])))
            res = res[["fac"]].reset_index()
            res.columns = ["code", "fac"]
            return res

        return full_run

    @kk.desktop_sender(title="嘿，分钟数据处理完啦～🎈")
    def get_daily_factors(
        self,
        func: Callable,
        fields: str = "*",
        chunksize: int = 10,
        show_time: bool = 0,
        tqdm_inside: bool = 0,
    ) -> None:
        """每次抽取chunksize天的截面上全部股票的分钟数据
        对每天的股票的数据计算因子值

        Parameters
        ----------
        func : Callable
            用于计算因子值的函数
        fields : str, optional
            股票数据涉及到哪些字段，排除不必要的字段，可以节约读取数据的时间，形如'date,code,num,close,amount,open'
            提取出的数据，自动按照code,date,num排序，因此code,date,num是必不可少的字段, by default "*"
        chunksize : int, optional
            每次读取的截面上的天数, by default 10
        show_time : bool, optional
            展示每次读取数据所需要的时间, by default 0
        tqdm_inside : bool, optional
            将进度条加在内部，而非外部，建议仅chunksize较大时使用, by default 0
        """
        if len(self.dates_new) > 0:
            for interval in self.dates_new_intervals:
                df = self.select_any_calculate(
                    dates=interval,
                    func=func,
                    fields=fields,
                    chunksize=chunksize,
                    show_time=show_time,
                    tqdm_inside=tqdm_inside,
                )
                self.factor_new.append(df)
            self.factor_new = pd.concat(self.factor_new)
            # 拼接新的和旧的
            self.factor = pd.concat([self.factor_old, self.factor_new]).sort_index()
            self.factor = self.factor.dropna(how="all")
            self.factor = self.factor.reset_index()
            self.factor = self.factor.rename(
                columns={list(self.factor.columns)[0]: "date"}
            )
            self.factor = self.factor.drop_duplicates(subset=["date"], keep="first")
            self.factor = self.factor.set_index("date")
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            # 存入本地
            self.factor.reset_index().to_feather(self.factor_file)
            logger.info(f"截止到{new_end_date}的因子值计算完了")
            # 删除存储在questdb的中途备份数据
            self.factor_steps.do_order(f"drop table {self.factor_file_pinyin}")
            logger.info("备份在questdb的表格已删除")

        else:
            self.factor = self.factor_old
            self.factor = self.factor.reset_index()
            self.factor = self.factor.rename(
                columns={list(self.factor.columns)[0]: "date"}
            )
            self.factor = self.factor.drop_duplicates(subset=["date"], keep="first")
            self.factor = self.factor.set_index("date")
            # 存入本地
            self.factor.reset_index().to_feather(self.factor_file)
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            logger.info(f"当前截止到{new_end_date}的因子值已经是最新的了")


class pure_fall_flexible(object):
    def __init__(
        self,
        factor_file: str,
        startdate: int = None,
        enddate: int = None,
        kind: str = "stock",
        clickhouse: bool = 0,
        questdb: bool = 0,
    ) -> None:
        """基于clickhouse的分钟数据，计算因子值，每天的因子值用到多日的数据，或者用到截面的数据
        对一段时间的截面数据进行操作，在get_daily_factors的func函数中
        请写入df=df.groupby([xxx]).apply(fff)之类的语句
        然后单独定义一个函数，作为要apply的fff，可以在apply上加进度条

        Parameters
        ----------
        factor_file : str
            用于存储因子的文件名称，请以'.feather'结尾
        startdate : int, optional
            计算因子的起始日期，形如20220816, by default None
        enddate : int, optional
            计算因子的终止日期，形如20220816, by default None
        kind : str, optional
            指定计算股票还是指数，指数则为'index', by default "stock"
        clickhouse : bool, optional
            使用clickhouse作为数据源，如果postgresql与本参数都为0，将依然从clickhouse中读取, by default 0
        questdb : bool, optional
            使用questdb作为数据源, by default 0
        """
        homeplace = HomePlace()
        self.kind = kind
        if clickhouse == 0 and questdb == 0:
            clickhouse = 1
        self.clickhouse = clickhouse
        self.questdb = questdb
        if clickhouse == 1:
            # 连接clickhouse
            self.chc = ClickHouseClient("minute_data")
        elif questdb:
            self.chc = Questdb()
        # 完整的因子文件路径
        factor_file = homeplace.factor_data_file + factor_file
        self.factor_file = factor_file
        # 读入之前的因子
        if os.path.exists(factor_file):
            factor_old = pd.read_feather(self.factor_file)
            factor_old.columns = ["date"] + list(factor_old.columns)[1:]
            factor_old = factor_old.drop_duplicates(subset=["date"])
            factor_old = factor_old.set_index("date")
            self.factor_old = factor_old
            # 已经算好的日子
            dates_old = sorted(list(factor_old.index.strftime("%Y%m%d").astype(int)))
            self.dates_old = dates_old
        else:
            self.factor_old = None
            self.dates_old = []
            logger.info("这个因子以前没有，正在重新计算")
        # 读取当前所有的日子
        dates_all = self.chc.show_all_dates(f"minute_data_{kind}")
        if startdate is None:
            ...
        else:
            dates_all = [i for i in dates_all if i >= startdate]
        if enddate is None:
            ...
        else:
            dates_all = [i for i in dates_all if i <= enddate]
        self.dates_all = dates_all
        # 需要新补充的日子
        self.dates_new = sorted([i for i in dates_all if i not in dates_old])

    def __call__(self) -> pd.DataFrame:
        """直接返回因子值的pd.DataFrame

        Returns
        -------
        `pd.DataFrame`
            计算出的因子值
        """
        return self.factor.copy()

    @kk.desktop_sender(title="嘿，分钟数据处理完啦～🎈")
    def get_daily_factors(
        self,
        func: Callable,
        fields: str = "*",
        chunksize: int = 250,
        show_time: bool = 0,
        tqdm_inside: bool = 0,
    ) -> None:
        """每次抽取chunksize天的截面上全部股票的分钟数据
        依照定义的函数计算因子值

        Parameters
        ----------
        func : Callable
            用于计算因子值的函数
        fields : str, optional
            股票数据涉及到哪些字段，排除不必要的字段，可以节约读取数据的时间，形如'date,code,num,close,amount,open'
            提取出的数据，自动按照code,date,num排序，因此code,date,num是必不可少的字段, by default "*"
        chunksize : int, optional
            每次读取的截面上的天数, by default 10
        show_time : bool, optional
            展示每次读取数据所需要的时间, by default 0
        tqdm_inside : bool, optional
            将进度条加在内部，而非外部，建议仅chunksize较大时使用, by default 0
        """
        the_func = partial(func)
        # 将需要更新的日子分块，每200天一组，一起运算
        dates_new_len = len(self.dates_new)
        if dates_new_len > 0:
            cut_points = list(range(0, dates_new_len, chunksize)) + [dates_new_len - 1]
            if cut_points[-1] == cut_points[-2]:
                cut_points = cut_points[:-1]
            self.cut_points = cut_points
            self.factor_new = []
            # 开始计算因子值
            if tqdm_inside:
                # 开始计算因子值
                for date1, date2 in cut_points:
                    if self.clickhouse == 1:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]*100} and date<={self.dates_new[date2]*100} order by code,date,num"
                    else:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]} and date<={self.dates_new[date2]}"
                    if show_time:
                        df = self.chc.get_data_show_time(sql_order)
                    else:
                        df = self.chc.get_data(sql_order)
                    if self.clickhouse == 1:
                        df = ((df.set_index("code")) / 100).reset_index()
                    if is_notebook():
                        tqdm.tqdm_notebook().pandas()
                    else:
                        tqdm.tqdm.pandas()
                    df = the_func(df)
                    if isinstance(df, pd.Series):
                        df = df.reset_index()
                    df.columns = ["date", "code", "fac"]
                    df = df.pivot(columns="code", index="date", values="fac")
                    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
                    self.factor_new.append(df)
            else:
                # 开始计算因子值
                for date1, date2 in tqdm.tqdm(cut_points, desc="不知乘月几人归，落月摇情满江树。"):
                    if self.clickhouse == 1:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]*100} and date<={self.dates_new[date2]*100} order by code,date,num"
                    else:
                        sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date>{self.dates_new[date1]} and date<={self.dates_new[date2]} order by code,date,num"
                    if show_time:
                        df = self.chc.get_data_show_time(sql_order)
                    else:
                        df = self.chc.get_data(sql_order)
                    if self.clickhouse == 1:
                        df = ((df.set_index("code")) / 100).reset_index()
                    df = the_func(df)
                    if isinstance(df, pd.Series):
                        df = df.reset_index()
                    df.columns = ["date", "code", "fac"]
                    df = df.pivot(columns="code", index="date", values="fac")
                    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
                    self.factor_new.append(df)
            self.factor_new = pd.concat(self.factor_new)
            # 拼接新的和旧的
            self.factor = pd.concat([self.factor_old, self.factor_new]).sort_index()
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            # 存入本地
            self.factor.reset_index().to_feather(self.factor_file)
            logger.info(f"截止到{new_end_date}的因子值计算完了")
        elif dates_new_len == 1:
            print("共1天")
            if tqdm_inside:
                # 开始计算因子值
                if self.clickhouse == 1:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]*100} order by code,date,num"
                else:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]} order by code,date,num"
                if show_time:
                    df = self.chc.get_data_show_time(sql_order)
                else:
                    df = self.chc.get_data(sql_order)
                if self.clickhouse == 1:
                    df = ((df.set_index("code")) / 100).reset_index()
                tqdm.tqdm.pandas()
                df = the_func(df)
                if isinstance(df, pd.Series):
                    df = df.reset_index()
                df.columns = ["date", "code", "fac"]
                df = df.pivot(columns="code", index="date", values="fac")
                df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
            else:
                # 开始计算因子值
                if self.clickhouse == 1:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]*100} order by code,date,num"
                else:
                    sql_order = f"select {fields} from minute_data.minute_data_{self.kind} where date={self.dates_new[0]} order by code,date,num"
                if show_time:
                    df = self.chc.get_data_show_time(sql_order)
                else:
                    df = self.chc.get_data(sql_order)
                if self.clickhouse == 1:
                    df = ((df.set_index("code")) / 100).reset_index()
                df = the_func(df)
                if isinstance(df, pd.Series):
                    df = df.reset_index()
                df.columns = ["date", "code", "fac"]
                df = df.pivot(columns="code", index="date", values="fac")
                df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
                self.factor_new.append(df)
            self.factor_new = df
            # 拼接新的和旧的
            self.factor = (
                pd.concat([self.factor_old, self.factor_new])
                .sort_index()
                .drop_duplicates()
            )
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            # 存入本地
            self.factor.reset_index().to_feather(self.factor_file)
            logger.info(f"补充{self.dates_new[0]}截止到{new_end_date}的因子值计算完了")
        else:
            self.factor = self.factor_old
            new_end_date = datetime.datetime.strftime(self.factor.index.max(), "%Y%m%d")
            logger.info(f"当前截止到{new_end_date}的因子值已经是最新的了")


class pure_coldwinter(object):
    # DONE: 可以自由添加其他要剔除的因子，或者替换某些要剔除的因子

    def __init__(
        self,
        facs_dict: dict = None,
        momentum: bool = 1,
        earningsyield: bool = 1,
        growth: bool = 1,
        liquidity: bool = 1,
        size: bool = 1,
        leverage: bool = 1,
        beta: bool = 1,
        nonlinearsize: bool = 1,
        residualvolatility: bool = 1,
        booktoprice: bool = 1,
    ) -> None:
        """读入10种常用风格因子，并可以额外加入其他因子

        Parameters
        ----------
        facs_dict : dict, optional
            额外加入的因子，名字为key，因子矩阵为value，形如`{'反转': ret20, '换手': tr20}`, by default None
        momentum : bool, optional
            是否删去动量因子, by default 1
        earningsyield : bool, optional
            是否删去盈利因子, by default 1
        growth : bool, optional
            是否删去成长因子, by default 1
        liquidity : bool, optional
            是否删去流动性因子, by default 1
        size : bool, optional
            是否删去规模因子, by default 1
        leverage : bool, optional
            是否删去杠杆因子, by default 1
        beta : bool, optional
            是否删去贝塔因子, by default 1
        nonlinearsize : bool, optional
            是否删去非线性市值因子, by default 1
        residualvolatility : bool, optional
            是否删去残差波动率因子, by default 1
        booktoprice : bool, optional
            是否删去账面市值比因子, by default 1
        """
        self.homeplace = HomePlace()
        # barra因子数据
        styles = os.listdir(self.homeplace.barra_data_file)
        styles = [i for i in styles if i.endswith(".feather")]
        barras = {}
        for s in styles:
            k = s.split(".")[0]
            v = pd.read_feather(self.homeplace.barra_data_file + s)
            v.columns = ["date"] + list(v.columns)[1:]
            v = v.set_index("date")
            barras[k] = v
        rename_dict = {
            "fac": "因子自身",
            "size": "市值",
            "nonlinearsize": "非线性市值",
            "booktoprice": "估值",
            "earningsyield": "盈利",
            "growth": "成长",
            "leverage": "杠杆",
            "liquidity": "流动性",
            "momentum": "动量",
            "residualvolatility": "波动率",
            "beta": "贝塔",
        }
        if momentum == 0:
            barras = {k: v for k, v in barras.items() if k != "momentum"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "momentum"}
        if earningsyield == 0:
            barras = {k: v for k, v in barras.items() if k != "earningsyield"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "earningsyield"}
        if growth == 0:
            barras = {k: v for k, v in barras.items() if k != "growth"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "growth"}
        if liquidity == 0:
            barras = {k: v for k, v in barras.items() if k != "liquidity"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "liquidity"}
        if size == 0:
            barras = {k: v for k, v in barras.items() if k != "size"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "size"}
        if leverage == 0:
            barras = {k: v for k, v in barras.items() if k != "leverage"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "leverage"}
        if beta == 0:
            barras = {k: v for k, v in barras.items() if k != "beta"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "beta"}
        if nonlinearsize == 0:
            barras = {k: v for k, v in barras.items() if k != "nonlinearsize"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "nonlinearsize"}
        if residualvolatility == 0:
            barras = {k: v for k, v in barras.items() if k != "residualvolatility"}
            rename_dict = {
                k: v for k, v in rename_dict.items() if k != "residualvolatility"
            }
        if booktoprice == 0:
            barras = {k: v for k, v in barras.items() if k != "booktoprice"}
            rename_dict = {k: v for k, v in rename_dict.items() if k != "booktoprice"}
        if facs_dict is not None:
            barras.update(facs_dict)
        self.barras = barras
        self.rename_dict = rename_dict
        sort_names = list(rename_dict.values())
        if facs_dict is not None:
            sort_names = sort_names + list(facs_dict.keys())
        sort_names = [i for i in sort_names if i != "因子自身"]
        self.sort_names = sort_names

    def __call__(self):
        """返回纯净因子值"""
        return self.snow_fac

    def set_factors_df_wide(self, df):
        """传入因子数据，时间为索引，代码为列名"""
        df1 = df.copy()
        # df1.index=df1.index-pd.DateOffset(months=1)
        df1 = df1.resample("M").last()
        df1 = df1.stack().reset_index()
        df1.columns = ["date", "code", "fac"]
        self.factors = df1.copy()

    def daily_to_monthly(self, df):
        """将日度的barra因子月度化"""
        df = df.resample("M").last()
        return df

    def get_monthly_barras_industrys(self):
        """将barra因子和行业哑变量变成月度数据"""
        for key, value in self.barras.items():
            self.barras[key] = self.daily_to_monthly(value)

    def wide_to_long(self, df, name):
        """将宽数据变成长数据，便于后续拼接"""
        df = df.stack().reset_index()
        df.columns = ["date", "code", name]
        df = df.set_index(["date", "code"])
        return df

    def get_wide_barras_industrys(self):
        """将barra因子和行业哑变量都变成长数据"""
        for key, value in self.barras.items():
            self.barras[key] = self.wide_to_long(value, key)

    def get_corr_pri_ols_pri(self):
        """拼接barra因子和行业哑变量，生成用于求相关系数和纯净因子的数据表"""
        if self.factors.shape[0] > 1:
            self.factors = self.factors.set_index(["date", "code"])
        self.corr_pri = pd.concat(
            [self.factors] + list(self.barras.values()), axis=1
        ).dropna()

    # DONE: 修改风格因子展示顺序至报告的顺序
    def get_corr(self):
        """计算每一期的相关系数，再求平均值"""
        self.corr_by_step = self.corr_pri.groupby(["date"]).apply(
            lambda x: x.corr().head(1)
        )
        self.__corr = self.corr_by_step.mean()
        self.__corr = self.__corr.rename(index=self.rename_dict)
        self.__corr = self.__corr.to_frame("相关系数").T

        self.__corr = self.__corr[self.sort_names]
        self.__corr = self.__corr.T

    @property
    def corr(self) -> pd.DataFrame:
        """因子和10个常用风格因子的相关系数

        Returns
        -------
        pd.DataFrame
            因子和10个常用风格因子的相关系数
        """
        return self.__corr.copy()

    def ols_in_group(self, df):
        """对每个时间段进行回归，并计算残差"""
        xs = list(df.columns)
        xs = [i for i in xs if i != "fac"]
        xs_join = "+".join(xs)
        ols_formula = "fac~" + xs_join
        ols_result = smf.ols(ols_formula, data=df).fit()
        ols_ws = {i: ols_result.params[i] for i in xs}
        ols_b = ols_result.params["Intercept"]
        to_minus = [ols_ws[i] * df[i] for i in xs]
        to_minus = reduce(lambda x, y: x + y, to_minus)
        df = df.assign(snow_fac=df.fac - to_minus - ols_b)
        df = df[["snow_fac"]]
        df = df.rename(columns={"snow_fac": "fac"})
        return df

    def get_snow_fac(self):
        """获得纯净因子"""
        self.snow_fac = self.corr_pri.groupby(["date"]).apply(self.ols_in_group)
        self.snow_fac = self.snow_fac.unstack()
        self.snow_fac.columns = list(map(lambda x: x[1], list(self.snow_fac.columns)))

    def run(self):
        """运行一些必要的函数"""
        self.get_monthly_barras_industrys()
        self.get_wide_barras_industrys()
        self.get_corr_pri_ols_pri()
        self.get_corr()
        self.get_snow_fac()


class pure_snowtrain(object):
    """直接返回纯净因子"""

    def __init__(
        self,
        factors: pd.DataFrame,
        facs_dict: dict = None,
        momentum: bool = 1,
        earningsyield: bool = 1,
        growth: bool = 1,
        liquidity: bool = 1,
        size: bool = 1,
        leverage: bool = 1,
        beta: bool = 1,
        nonlinearsize: bool = 1,
        residualvolatility: bool = 1,
        booktoprice: bool = 1,
    ) -> None:
        """计算因子值与10种常用风格因子之间的相关性，并进行纯净化，可以额外加入其他因子

        Parameters
        ----------
        factors : pd.DataFrame
            要考察的因子值，index为时间，columns为股票代码，values为因子值
        facs_dict : dict, optional
            _description_, by default None
        momentum : bool, optional
            _description_, by default 1
        earningsyield : bool, optional
            _description_, by default 1
        growth : bool, optional
            _description_, by default 1
        liquidity : bool, optional
            _description_, by default 1
        size : bool, optional
            _description_, by default 1
        leverage : bool, optional
            _description_, by default 1
        beta : bool, optional
            _description_, by default 1
        nonlinearsize : bool, optional
            _description_, by default 1
        residualvolatility : bool, optional
            _description_, by default 1
        booktoprice : bool, optional
            _description_, by default 1
        """
        self.winter = pure_coldwinter(
            facs_dict=facs_dict,
            momentum=momentum,
            earningsyield=earningsyield,
            growth=growth,
            liquidity=liquidity,
            size=size,
            leverage=leverage,
            beta=beta,
            nonlinearsize=nonlinearsize,
            residualvolatility=residualvolatility,
            booktoprice=booktoprice,
        )
        self.winter.set_factors_df_wide(factors.copy())
        self.winter.run()
        self.corr = self.winter.corr

    def __call__(self) -> pd.DataFrame:
        """获得纯净化之后的因子值

        Returns
        -------
        pd.DataFrame
            纯净化之后的因子值
        """
        return self.winter.snow_fac.copy()


class pure_newyear(object):
    """转为生成25分组和百分组的收益矩阵而封装"""

    def __init__(
        self,
        facx: pd.DataFrame,
        facy: pd.DataFrame,
        group_num_single: int,
        namex: str = "主",
        namey: str = "次",
    ) -> None:
        """条件双变量排序法，先对所有股票，依照因子facx进行排序
        然后在每个组内，依照facy进行排序，最后统计各个组内的平均收益率

        Parameters
        ----------
        facx : pd.DataFrame
            首先进行排序的因子，通常为控制变量，相当于正交化中的自变量
            index为时间，columns为股票代码，values为因子值
        facy : pd.DataFrame
            在facx的各个组内，依照facy进行排序，为主要要研究的因子
            index为时间，columns为股票代码，values为因子值
        group_num_single : int
            单个因子分成几组，通常为5或10
        namex : str, optional
            facx这一因子的名字, by default "主"
        namey : str, optional
            facy这一因子的名字, by default "次"
        """
        homex = pure_fallmount(facx)
        homey = pure_fallmount(facy)
        if group_num_single == 5:
            homexy = homex > homey
        elif group_num_single == 10:
            homexy = homex >> homey
        shen = pure_moonnight(
            homexy(), group_num_single**2, plt_plot=False, print_comments=False
        )
        sq = shen.shen.square_rets.copy()
        sq.index = [namex + str(i) for i in list(sq.index)]
        sq.columns = [namey + str(i) for i in list(sq.columns)]
        self.square_rets = sq

    def __call__(self) -> pd.DataFrame:
        """调用对象时，返回最终结果，正方形的分组年化收益率表

        Returns
        -------
        `pd.DataFrame`
            每个组的年化收益率
        """
        return self.square_rets.copy()


class pure_dawn(object):
    """
    因子切割论的母框架，可以对两个因子进行类似于因子切割的操作
    可用于派生任何"以两个因子生成一个因子"的子类
    使用举例
    cut函数里，必须带有输入变量df,df有两个columns，一个名为'fac1'，一个名为'fac2'，df是最近一个回看期内的数据
    ```python
    class Cut(pure_dawn):

    def cut(self,df):
        df=df.sort_values('fac1')
        df=df.assign(fac3=df.fac1*df.fac2)
        ret0=df.fac2.iloc[:4].mean()
        ret1=df.fac2.iloc[4:8].mean()
        ret2=df.fac2.iloc[8:12].mean()
        ret3=df.fac2.iloc[12:16].mean()
        ret4=df.fac2.iloc[16:].mean()
        aret0=df.fac3.iloc[:4].mean()
        aret1=df.fac3.iloc[4:8].mean()
        aret2=df.fac3.iloc[8:12].mean()
        aret3=df.fac3.iloc[12:16].mean()
        aret4=df.fac3.iloc[16:].mean()
        return ret0,ret1,ret2,ret3,ret4,aret0,aret1,aret2,aret3,aret4

    cut=Cut(ct,ret_inday)
    cut.run(cut.cut)

    cut0=get_value(cut(),0)
    cut1=get_value(cut(),1)
    cut2=get_value(cut(),2)
    cut3=get_value(cut(),3)
    cut4=get_value(cut(),4)
    ```
    """

    def __init__(self, fac1: pd.DataFrame, fac2: pd.DataFrame, *args: list) -> None:
        """几个因子的操作，每个月操作一次

        Parameters
        ----------
        fac1 : pd.DataFrame
            因子值1，index为时间，columns为股票代码，values为因子值
        fac2 : pd.DataFrame
            因子2，index为时间，columns为股票代码，values为因子值
        """
        self.fac1 = fac1
        self.fac1 = self.fac1.stack().reset_index()
        self.fac1.columns = ["date", "code", "fac1"]
        self.fac2 = fac2
        self.fac2 = self.fac2.stack().reset_index()
        self.fac2.columns = ["date", "code", "fac2"]
        fac_all = pd.merge(self.fac1, self.fac2, on=["date", "code"])
        for i, fac in enumerate(args):
            fac = fac.stack().reset_index()
            fac.columns = ["date", "code", f"fac{i+3}"]
            fac_all = pd.merge(fac_all, fac, on=["date", "code"])
        fac_all = fac_all.sort_values(["date", "code"])
        self.fac = fac_all.copy()

    def __call__(self) -> pd.DataFrame:
        """返回最终月度因子值

        Returns
        -------
        `pd.DataFrame`
            最终因子值
        """
        return self.fac.copy()

    def get_fac_long_and_tradedays(self):
        """将两个因子的矩阵转化为长列表"""
        self.tradedays = sorted(list(set(self.fac.date)))

    def get_month_starts_and_ends(self, backsee=20):
        """计算出每个月回看期间的起点日和终点日"""
        self.month_ends = [
            i
            for i, j in zip(self.tradedays[:-1], self.tradedays[1:])
            if i.month != j.month
        ]
        self.month_ends.append(self.tradedays[-1])
        self.month_starts = [
            self.find_begin(self.tradedays, i, backsee=backsee) for i in self.month_ends
        ]
        self.month_starts[0] = self.tradedays[0]

    def find_begin(self, tradedays, end_day, backsee=20):
        """找出回看若干天的开始日，默认为20"""
        end_day_index = tradedays.index(end_day)
        start_day_index = end_day_index - backsee + 1
        start_day = tradedays[start_day_index]
        return start_day

    def make_monthly_factors_single_code(self, df, func):
        """
        对单一股票来计算月度因子
        func为单月执行的函数，返回值应为月度因子，如一个float或一个list
        df为一个股票的四列表，包含时间、代码、因子1和因子2
        """
        res = {}
        for start, end in zip(self.month_starts, self.month_ends):
            this_month = df[(df.date >= start) & (df.date <= end)]
            res[end] = func(this_month)
        dates = list(res.keys())
        corrs = list(res.values())
        part = pd.DataFrame({"date": dates, "corr": corrs})
        return part

    def get_monthly_factor(self, func):
        """运行自己写的函数，获得月度因子"""
        if is_notebook():
            tqdm.tqdm_notebook().pandas()
        else:
            tqdm.tqdm.pandas(desc="when the dawn comes, tonight will be a memory too.")
        self.fac = self.fac.groupby(["code"]).progress_apply(
            lambda x: self.make_monthly_factors_single_code(x, func)
        )
        self.fac = (
            self.fac.reset_index(level=1, drop=True)
            .reset_index()
            .set_index(["date", "code"])
            .unstack()
        )
        self.fac.columns = [i[1] for i in list(self.fac.columns)]
        self.fac = self.fac.resample("M").last()

    @kk.desktop_sender(title="嘿，切割完成啦🛁")
    def run(self, func: Callable, backsee: int = 20) -> None:
        """执行计算的框架，产生因子值

        Parameters
        ----------
        func : Callable
            每个月要进行的操作
        backsee : int, optional
            回看期，即每个月月底对过去多少天进行计算, by default 20
        """
        self.get_fac_long_and_tradedays()
        self.get_month_starts_and_ends(backsee=backsee)
        self.get_monthly_factor(func)


def follow_tests(
    fac: pd.DataFrame,
    comments_writer: pd.ExcelWriter = None,
    net_values_writer: pd.ExcelWriter = None,
    pos: bool = 0,
    neg: bool = 0,
    swindustry: bool = 0,
    zxindustry: bool = 0,
    nums: list[int] = [3],
):
    """因子完成全A测试后，进行的一些必要的后续测试，包括各个分组表现、相关系数与纯净化、3510的多空和多头、各个行业Rank IC、各个行业买3只超额表现

    Parameters
    ----------
    fac : pd.DataFrame
        要进行后续测试的因子值，index是时间，columns是股票代码，values是因子值
    comments_writer : pd.ExcelWriter, optional
        写入评价指标的excel, by default None
    net_values_writer : pd.ExcelWriter, optional
        写入净值序列的excel, by default None
    pos : bool, optional
        因子的方向为正, by default 0
    neg : bool, optional
        因子的方向为负, by default 0
    swindustry : bool, optional
        使用申万一级行业, by default 0
    zxindustry : bool, optional
        使用中信一级行业, by default 0
    nums : list[int], optional
        各个行业买几只股票, by default [3]

    Raises
    ------
    IOError
        如果未指定因子正负方向，将报错
    """
    if comments_writer is None:
        from pure_ocean_breeze.legacy_version.v3p4.state.states import COMMENTS_WRITER

        comments_writer = COMMENTS_WRITER
    if net_values_writer is None:
        from pure_ocean_breeze.legacy_version.v3p4.state.states import NET_VALUES_WRITER

        net_values_writer = NET_VALUES_WRITER

    shen = pure_moonnight(fac)
    shen.comments_ten().to_excel(comments_writer, sheet_name="十分组")
    """相关系数与纯净化"""
    pure_fac = pure_snowtrain(fac)
    pure_fac.corr.to_excel(comments_writer, sheet_name="相关系数")
    shen = pure_moonnight(
        pure_fac(),
        comments_writer=comments_writer,
        net_values_writer=net_values_writer,
        sheetname="纯净",
    )
    """3510多空和多头"""
    # 300
    fi300 = daily_factor_on300500(fac, hs300=1)
    shen = pure_moonnight(
        fi300,
        comments_writer=comments_writer,
        net_values_writer=net_values_writer,
        sheetname="300多空",
    )
    if pos:
        make_relative_comments(shen.shen.group_rets.group10, hs300=1).to_excel(
            comments_writer, sheet_name="300超额"
        )
        make_relative_comments_plot(shen.shen.group_rets.group10, hs300=1).to_excel(
            net_values_writer, sheet_name="300超额"
        )
    elif neg:
        make_relative_comments(shen.shen.group_rets.group1, hs300=1).to_excel(
            comments_writer, sheet_name="300超额"
        )
        make_relative_comments_plot(shen.shen.group_rets.group1, hs300=1).to_excel(
            net_values_writer, sheet_name="300超额"
        )
    else:
        raise IOError("请指定因子的方向是正是负🤒")
    # 500
    fi500 = daily_factor_on300500(fac, zz500=1)
    shen = pure_moonnight(
        fi500,
        comments_writer=comments_writer,
        net_values_writer=net_values_writer,
        sheetname="500多空",
    )
    if pos:
        make_relative_comments(shen.shen.group_rets.group10, zz500=1).to_excel(
            comments_writer, sheet_name="500超额"
        )
        make_relative_comments_plot(shen.shen.group_rets.group10, zz500=1).to_excel(
            net_values_writer, sheet_name="500超额"
        )
    else:
        make_relative_comments(shen.shen.group_rets.group1, zz500=1).to_excel(
            comments_writer, sheet_name="500超额"
        )
        make_relative_comments_plot(shen.shen.group_rets.group1, zz500=1).to_excel(
            net_values_writer, sheet_name="500超额"
        )
    # 1000
    fi1000 = daily_factor_on300500(fac, zz1000=1)
    shen = pure_moonnight(
        fi1000,
        comments_writer=comments_writer,
        net_values_writer=net_values_writer,
        sheetname="1000多空",
    )
    if pos:
        make_relative_comments(shen.shen.group_rets.group10, zz1000=1).to_excel(
            comments_writer, sheet_name="1000超额"
        )
        make_relative_comments_plot(shen.shen.group_rets.group10, zz1000=1).to_excel(
            net_values_writer, sheet_name="1000超额"
        )
    else:
        make_relative_comments(shen.shen.group_rets.group1, zz1000=1).to_excel(
            comments_writer, sheet_name="1000超额"
        )
        make_relative_comments_plot(shen.shen.group_rets.group1, zz1000=1).to_excel(
            net_values_writer, sheet_name="1000超额"
        )
    # 各行业Rank IC
    rankics = rankic_test_on_industry(fac, comments_writer)
    # 买3只超额表现
    rets = long_test_on_industry(
        fac, nums, pos=pos, neg=neg, swindustry=swindustry, zxindustry=zxindustry
    )
    logger.success("因子后续的必要测试全部完成")


class pure_helper(object):
    def __init__(
        self,
        df_main: pd.DataFrame,
        df_helper: pd.DataFrame,
        func: Callable = None,
        group: int = 10,
    ) -> None:
        """使用因子b的值大小，对因子a进行分组，并可以在组内进行某种操作

        Parameters
        ----------
        df_main : pd.DataFrame
            要被分组并进行操作的因子
        df_helper : pd.DataFrame
            用来做分组的依据
        func : Callable, optional
            分组后，组内要进行的操作, by default None
        group : int, optional
            要分的组数, by default 10
        """
        self.df_main = df_main
        self.df_helper = df_helper
        self.func = func
        self.group = group
        if self.func is None:
            self.__data = self.sort_a_with_b()
        else:
            self.__data = self.sort_a_with_b_func()

    @property
    def data(self):
        return self.__data

    def __call__(self) -> pd.DataFrame:
        return self.data

    def sort_a_with_b(self):
        dfb = to_group(self.df_helper, group=self.group)
        dfb = dfb.stack().reset_index()
        dfb.columns = ["date", "code", "group"]
        dfa = self.df_main.stack().reset_index()
        dfa.columns = ["date", "code", "target"]
        df = pd.merge(dfa, dfb, on=["date", "code"])
        return df

    def sort_a_with_b_func(self):
        the_func = partial(self.func)
        df = self.sort_a_with_b().drop(columns=["code"])
        df = (
            df.groupby(["date", "group"])
            .apply(the_func)
            .drop(columns=["group"])
            .reset_index()
        )
        df = df.pivot(index="date", columns="group", values="target")
        df.columns = [f"group{str(int(i+1))}" for i in list(df.columns)]
        return df


class pure_fama(object):
    # @lru_cache(maxsize=None)
    def __init__(
        self,
        factors: list[pd.DataFrame],
        minus_group: Union[list, float] = 3,
        backsee: int = 20,
        rets: pd.DataFrame = None,
        value_weighted: bool = 1,
        add_market: bool = 1,
        add_market_series: pd.Series = None,
        factors_names: list = None,
        betas_rets: bool = 0,
    ) -> None:
        """使用fama三因子的方法，将个股的收益率，拆分出各个因子带来的收益率以及特质的收益率
        分别计算每一期，各个因子收益率的值，超额收益率，因子的暴露，以及特质收益率

        Parameters
        ----------
        factors : list[pd.DataFrame]
            用于解释收益的各个因子值，每一个都是index为时间，columns为股票代码，values为因子值的dataframe
        minus_group : Union[list, float], optional
            每一个因子将截面上的股票分为几组, by default 3
        backsee : int, optional
            做时序回归时，回看的天数, by default 20
        rets : pd.DataFrame, optional
            每只个股的收益率，index为时间，columns为股票代码，values为收益率，默认使用当日日间收益率, by default None
        value_weighted : bool, optional
            是否使用流通市值加权, by default 1
        add_market : bool, optional
            是否加入市场收益率因子，默认使用中证全指的每日日间收益率, by default 1
        add_market_series : bool, optional
            加入的市场收益率的数据，如果没指定，则使用中证全指的日间收益率, by default None
        factors_names : list, optional
            各个因子的名字，默认为fac0(市场收益率因子，如果没有，则从fac1开始),fac1,fac2,fac3, by default None
        betas_rets : bool, optional
            是否计算每只个股的由于暴露在每个因子上所带来的收益率, by default 0
        """
        start = max(
            [int(datetime.datetime.strftime(i.index.min(), "%Y%m%d")) for i in factors]
        )
        self.backsee = backsee
        self.factors = factors
        self.factors_names = factors_names
        if isinstance(minus_group, int):
            minus_group = [minus_group] * len(factors)
        self.minus_group = minus_group
        if rets is None:
            closes = read_daily(close=1, start=start)
            rets = closes / closes.shift(1) - 1
        self.rets = rets
        self.factors_group = [
            to_group(i, group=j) for i, j in zip(self.factors, self.minus_group)
        ]
        self.factors_group_long = [(i == 0) + 0 for i in self.factors_group]
        self.factors_group_short = [
            (i == (j - 1)) + 0 for i, j in zip(self.factors_group, self.minus_group)
        ]
        self.value_weighted = value_weighted
        if value_weighted:
            self.cap = read_daily(flow_cap=1, start=start)
            self.factors_group_long = [self.cap * i for i in self.factors_group_long]
            self.factors_group_short = [self.cap * i for i in self.factors_group_short]
            self.factors_group_long = [
                (i.T / i.T.sum()).T for i in self.factors_group_long
            ]
            self.factors_group_short = [
                (i.T / i.T.sum()).T for i in self.factors_group_short
            ]
            self.factors_rets_long = [
                (self.rets * i).sum(axis=1).to_frame(f"fac{num+1}")
                for num, i in enumerate(self.factors_group_long)
            ]
            self.factors_rets_short = [
                (self.rets * i).sum(axis=1).to_frame(f"fac{num+1}")
                for num, i in enumerate(self.factors_group_short)
            ]
        else:
            self.factors_rets_long = [
                (self.rets * i).mean(axis=1).to_frame(f"fac{num+1}")
                for num, i in enumerate(self.factors_group_long)
            ]
            self.factors_rets_short = [
                (self.rets * i).mean(axis=1).to_frame(f"fac{num+1}")
                for num, i in enumerate(self.factors_group_short)
            ]
        self.rets_long = pd.concat(self.factors_rets_long, axis=1)
        self.rets_short = pd.concat(self.factors_rets_short, axis=1)
        self.__factors_rets = self.rets_long - self.rets_short
        if add_market_series is not None:
            add_market = 1
        self.add_market = add_market
        if add_market:
            if add_market_series is None:
                closes = read_market(close=1, every_stock=0, start=start).to_frame(
                    "fac0"
                )
            else:
                closes = add_market_series.to_frame("fac0")
            rets = closes / closes.shift(1) - 1
            self.__factors_rets = pd.concat([rets, self.__factors_rets], axis=1)
            if factors_names is not None:
                factors_names = ["市场"] + factors_names
        self.__data = self.make_df(self.rets, self.__factors_rets)
        if is_notebook():
            tqdm.tqdm_notebook().pandas()
        else:
            tqdm.tqdm.pandas()
        self.__coefficients = (
            self.__data.groupby("code").progress_apply(self.ols_in).reset_index()
        )
        self.__coefficients = self.__coefficients.rename(
            columns={
                i: "co" + i for i in list(self.__coefficients.columns) if "fac" in i
            }
        )
        self.__data = pd.merge(
            self.__data.reset_index(), self.__coefficients, on=["date", "code"]
        )
        betas = [
            self.__data[i] * self.__data["co" + i]
            for i in list(self.__data.columns)
            if i.startswith("fac")
        ]
        betas = sum(betas)
        self.__data = self.__data.assign(
            idiosyncratic=self.__data.ret - self.__data.intercept - betas
        )
        self.__idiosyncratic = self.__data.pivot(
            index="date", columns="code", values="idiosyncratic"
        )
        self.__alphas = self.__data.pivot(
            index="date", columns="code", values="intercept"
        )
        if factors_names is None:
            self.__betas = {
                i: self.__data.pivot(index="date", columns="code", values=i)
                for i in list(self.__data.columns)
                if i.startswith("fac")
            }
        else:
            facs = [i for i in list(self.__data.columns) if i.startswith("fac")]
            self.__betas = {
                factors_names[num]: self.__data.pivot(
                    index="date", columns="code", values=i
                )
                for num, i in enumerate(facs)
            }
        if betas_rets:
            if add_market:
                if add_market_series is None:
                    factors = [read_market(close=1, start=start)] + factors
                else:
                    factors = [
                        pd.DataFrame(
                            {k: add_market_series for k in list(factors[0].columns)},
                            index=factors[0].index,
                        )
                    ] + factors
            self.__betas_rets = {
                d1[0]: d1[1] * d2 for d1, d2 in zip(self.__betas, factors)
            }
        else:
            self.__betas_rets = "您如果想计算各个股票在各个因子的收益率，请先指定betas_rets参数为True"

    @property
    def idiosyncratic(self):
        return self.__idiosyncratic

    @property
    def data(self):
        return self.__data

    @property
    def alphas(self):
        return self.__alphas

    @property
    def betas(self):
        return self.__betas

    @property
    def betas_rets(self):
        return self.__betas_rets

    @property
    def factors_rets(self):
        return self.__factors_rets

    @property
    def coefficients(self):
        return self.__coefficients

    def __call__(self):
        return self.idiosyncratic

    def make_df(self, rets, facs):
        rets = rets.stack().reset_index()
        rets.columns = ["date", "code", "ret"]
        facs = facs.reset_index()
        facs.columns = ["date"] + list(facs.columns)[1:]
        df = pd.merge(rets, facs, on=["date"])
        df = df.set_index("date")
        return df

    def ols_in(self, df):
        try:
            if self.add_market:
                x = df[["fac0"] + [f"fac{i+1}" for i in range(len(self.factors))]]
            else:
                x = df[[f"fac{i+1}" for i in range(len(self.factors))]]
            ols = po.PandasRollingOLS(
                y=df[["ret"]],
                x=x,
                window=self.backsee,
            )
            betas = ols.beta
            alpha = ols.alpha
            return pd.concat([alpha, betas], axis=1)
        except Exception:
            # 有些数据总共不足，那就跳过
            ...


class pure_rollingols(object):
    def __init__(
        self,
        y: pd.DataFrame,
        xs: Union[list[pd.DataFrame], pd.DataFrame],
        backsee: int = 20,
        factors_names: list[str] = None,
    ) -> None:
        """使用若干个dataframe，对应的股票进行指定窗口的时序滚动回归

        Parameters
        ----------
        y : pd.DataFrame
            滚动回归中的因变量y，index是时间，columns是股票代码
        xs : Union[list[pd.DataFrame], pd.DataFrame]
            滚动回归中的自变量xi，每一个dataframe，index是时间，columns是股票代码
        backsee : int, optional
            滚动回归的时间窗口, by default 20
        factors_names : list[str], optional
            xs中，每个因子的名字, by default None
        """
        self.backsee = backsee
        self.y = y
        if not isinstance(xs, list):
            xs = [xs]
        self.xs = xs
        y = y.stack().reset_index()
        xs = [i.stack().reset_index() for i in xs]
        y.columns = ["date", "code", "y"]
        xs = [
            i.rename(
                columns={list(i.columns)[1]: "code", list(i.columns)[2]: f"x{j+1}"}
            )
            for j, i in enumerate(xs)
        ]
        xs = [y] + xs
        xs = reduce(lambda x, y: pd.merge(x, y, on=["date", "code"]), xs)
        xs = xs.set_index("date")
        self.__data = xs
        self.haha = xs
        if is_notebook():
            tqdm.tqdm_notebook().pandas()
        else:
            tqdm.tqdm.pandas()
        self.__coefficients = (
            self.__data.groupby("code").progress_apply(self.ols_in).reset_index()
        )
        self.__coefficients = self.__coefficients.rename(
            columns={i: "co" + i for i in list(self.__coefficients.columns) if "x" in i}
        )
        self.__data = pd.merge(
            self.__data.reset_index(), self.__coefficients, on=["date", "code"]
        )
        betas = [
            self.__data[i] * self.__data["co" + i]
            for i in list(self.__data.columns)
            if i.startswith("x")
        ]
        betas = sum(betas)
        self.__data = self.__data.assign(
            residual=self.__data.y - self.__data.intercept - betas
        )
        self.__residual = self.__data.pivot(
            index="date", columns="code", values="residual"
        )
        self.__alphas = self.__data.pivot(
            index="date", columns="code", values="intercept"
        )
        if factors_names is None:
            self.__betas = {
                i: self.__data.pivot(index="date", columns="code", values=i)
                for i in list(self.__data.columns)
                if i.startswith("x")
            }
        else:
            facs = [i for i in list(self.__data.columns) if i.startswith("x")]
            self.__betas = {
                factors_names[num]: self.__data.pivot(
                    index="date", columns="code", values=i
                )
                for num, i in enumerate(facs)
            }
        if len(list(self.__betas)) == 1:
            self.__betas = list(self.__betas.values())[0]

    @property
    def residual(self):
        return self.__residual

    @property
    def data(self):
        return self.__data

    @property
    def alphas(self):
        return self.__alphas

    @property
    def betas(self):
        return self.__betas

    @property
    def coefficients(self):
        return self.__coefficients

    def ols_in(self, df):
        try:
            ols = po.PandasRollingOLS(
                y=df[["y"]],
                x=df[[f"x{i+1}" for i in range(len(self.xs))]],
                window=self.backsee,
            )
            betas = ols.beta
            alpha = ols.alpha
            return pd.concat([alpha, betas], axis=1)

        except Exception:
            # 有些数据总共不足，那就跳过
            ...


def test_on_300500(
    df: pd.DataFrame,
    hs300: bool = 0,
    zz500: bool = 0,
    zz1000: bool = 0,
    gz2000: bool = 0,
    iplot: bool = 1,
) -> pd.Series:
    """对因子在指数成分股内进行多空和多头测试

    Parameters
    ----------
    df : pd.DataFrame
        因子值，index为时间，columns为股票代码
    hs300 : bool, optional
        在沪深300成分股内测试, by default 0
    zz500 : bool, optional
        在中证500成分股内测试, by default 0
    zz1000 : bool, optional
        在中证1000成分股内测试, by default 0
    gz1000 : bool, optional
        在国证2000成分股内测试, by default 0
    iplot : bol,optional
        多空回测的时候，是否使用cufflinks绘画

    Returns
    -------
    pd.Series
        多头组在该指数上的超额收益序列
    """
    fi300 = daily_factor_on300500(
        df, hs300=hs300, zz500=zz500, zz1000=zz1000, gz2000=gz2000
    )
    shen = pure_moonnight(fi300, iplot=iplot)
    if (
        shen.shen.group_net_values.group1.iloc[-1]
        > shen.shen.group_net_values.group10.iloc[-1]
    ):
        print(
            make_relative_comments(
                shen.shen.group_rets.group1,
                hs300=hs300,
                zz500=zz500,
                zz1000=zz1000,
                gz2000=gz2000,
            )
        )
        abrets = make_relative_comments_plot(
            shen.shen.group_rets.group1,
            hs300=hs300,
            zz500=zz500,
            zz1000=zz1000,
            gz2000=gz2000,
        )
        return abrets
    else:
        print(
            make_relative_comments(
                shen.shen.group_rets.group10,
                hs300=hs300,
                zz500=zz500,
                zz1000=zz1000,
                gz2000=gz2000,
            )
        )
        abrets = make_relative_comments_plot(
            shen.shen.group_rets.group10,
            hs300=hs300,
            zz500=zz500,
            zz1000=zz1000,
            gz2000=gz2000,
        )
        return abrets


def test_on_index_four(
    df: pd.DataFrame, iplot: bool = 1, gz2000: bool = 0, boxcox: bool = 1
) -> pd.DataFrame:
    """对因子同时在沪深300、中证500、中证1000、国证2000这4个指数成分股内进行多空和多头超额测试

    Parameters
    ----------
    df : pd.DataFrame
        因子值，index为时间，columns为股票代码
    iplot : bol,optional
        多空回测的时候，是否使用cufflinks绘画
    gz2000 : bool, optional
        是否进行国证2000上的测试, by default 0
    boxcox : bool, optional
        是否进行行业市值中性化处理, by default 1

    Returns
    -------
    pd.DataFrame
        多头组在各个指数上的超额收益序列
    """
    fi300 = daily_factor_on300500(df, hs300=1)
    shen = pure_moonnight(fi300, iplot=iplot, boxcox=boxcox)
    if (
        shen.shen.group_net_values.group1.iloc[-1]
        > shen.shen.group_net_values.group10.iloc[-1]
    ):
        com300, net300 = make_relative_comments(
            shen.shen.group_rets.group1, hs300=1, show_nets=1
        )
        fi500 = daily_factor_on300500(df, zz500=1)
        shen = pure_moonnight(fi500, iplot=iplot, boxcox=boxcox)
        com500, net500 = make_relative_comments(
            shen.shen.group_rets.group1, zz500=1, show_nets=1
        )
        fi1000 = daily_factor_on300500(df, zz1000=1)
        shen = pure_moonnight(fi1000, iplot=iplot, boxcox=boxcox)
        com1000, net1000 = make_relative_comments(
            shen.shen.group_rets.group1, zz1000=1, show_nets=1
        )
        if gz2000:
            fi2000 = daily_factor_on300500(df, gz2000=1)
            shen = pure_moonnight(fi2000, iplot=iplot, boxcox=boxcox)
            com2000, net2000 = make_relative_comments(
                shen.shen.group_rets.group1, gz2000=1, show_nets=1
            )
    else:
        com300, net300 = make_relative_comments(
            shen.shen.group_rets.group10, hs300=1, show_nets=1
        )
        fi500 = daily_factor_on300500(df, zz500=1)
        shen = pure_moonnight(fi500, iplot=iplot, boxcox=boxcox)
        com500, net500 = make_relative_comments(
            shen.shen.group_rets.group10, zz500=1, show_nets=1
        )
        fi1000 = daily_factor_on300500(df, zz1000=1)
        shen = pure_moonnight(fi1000, iplot=iplot, boxcox=boxcox)
        com1000, net1000 = make_relative_comments(
            shen.shen.group_rets.group10, zz1000=1, show_nets=1
        )
        if gz2000:
            fi2000 = daily_factor_on300500(df, gz2000=1)
            shen = pure_moonnight(fi2000, iplot=iplot, boxcox=boxcox)
            com2000, net2000 = make_relative_comments(
                shen.shen.group_rets.group10, gz2000=1, show_nets=1
            )
    com300 = com300.to_frame("300超额")
    com500 = com500.to_frame("500超额")
    com1000 = com1000.to_frame("1000超额")
    if gz2000:
        com2000 = com2000.to_frame("2000超额")
        coms = pd.concat([com300, com500, com1000, com2000], axis=1)
    else:
        coms = pd.concat([com300, com500, com1000], axis=1)
    coms = np.around(coms, 3)
    if gz2000:
        nets = pd.concat([net300, net500, net1000, net2000], axis=1)
        nets.columns = ["300超额", "500超额", "1000超额", "2000超额"]
    else:
        nets = pd.concat([net300, net500, net1000], axis=1)
        nets.columns = ["300超额", "500超额", "1000超额"]
    coms = coms.reset_index()
    if iplot:
        figs = cf.figures(
            nets,
            [dict(kind="line", y=list(nets.columns))],
            asList=True,
        )
        coms = coms.rename(columns={list(coms)[0]: "绩效指标"})
        table = FF.create_table(coms.iloc[::-1])
        table.update_yaxes(matches=None)
        figs.append(table)
        figs = [figs[-1]] + figs[:-1]
        figs[1].update_layout(
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        base_layout = cf.tools.get_base_layout(figs)
        if gz2000:
            sp = cf.subplots(
                figs,
                shape=(2, 10),
                base_layout=base_layout,
                vertical_spacing=0.15,
                horizontal_spacing=0.03,
                shared_yaxes=False,
                specs=[
                    [
                        None,
                        {"rowspan": 2, "colspan": 4},
                        None,
                        None,
                        None,
                        {"rowspan": 2, "colspan": 5},
                        None,
                        None,
                        None,
                        None,
                    ],
                    [
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                ],
            )
        else:
            sp = cf.subplots(
                figs,
                shape=(2, 10),
                base_layout=base_layout,
                vertical_spacing=0.15,
                horizontal_spacing=0.03,
                shared_yaxes=False,
                specs=[
                    [
                        None,
                        {"rowspan": 2, "colspan": 3},
                        None,
                        None,
                        {"rowspan": 2, "colspan": 6},
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                    [
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                ],
            )
        sp["layout"].update(showlegend=True)
        cf.iplot(sp)
    else:
        nets.plot()
        plt.show()
        tb = Texttable()
        tb.set_cols_width([8] + [7] + [8] * 2 + [7] * 2 + [8])
        tb.set_cols_dtype(["f"] * 7)
        tb.header(list(coms.T.reset_index().columns))
        tb.add_rows(coms.T.reset_index().to_numpy(), header=True)
        print(tb.draw())


class pure_star(object):
    def __init__(
        self,
        fac: pd.Series,
        code: str = None,
        price_opens: pd.Series = None,
        iplot: bool = 1,
        comments_writer: pd.ExcelWriter = None,
        net_values_writer: pd.ExcelWriter = None,
        sheetname: str = None,
    ):
        """择时回测框架，输入仓位比例或信号值，依据信号买入对应的股票或指数，并考察绝对收益、超额收益和基准收益
        回测方式为，t日收盘时获得信号，t+1日开盘时以开盘价买入，t+2开盘时以开盘价卖出

        Parameters
        ----------
        fac : pd.Series
            仓位比例序列，或信号序列，输入信号序列时即为0和1，输入仓位比例时，将每一期的收益按照对应比例缩小
        code : str, optional
            回测的资产代码，可以为股票代码或基金代码, by default None
        price_opens : pd.Series, optional
            资产的开盘价序列, by default None
        iplot : bool, optional
            使用cufflinks呈现回测绩效和走势图, by default 1
        comments_writer : pd.ExcelWriter, optional
            绩效评价的存储文件, by default None
        net_values_writer : pd.ExcelWriter, optional
            净值序列的存储文件, by default None
        sheetname : str, optional
            存储文件的工作表的名字, by default None
        """
        if code is not None:
            x1 = code.split(".")[0]
            x2 = code.split(".")[1]
            if (x1[0] == "0" or x1[:2] == "30") and x2 == "SZ":
                kind = "stock"
            elif x1[0] == "6" and x2 == "SH":
                kind = "stock"
            else:
                kind = "index"
            self.kind = kind
            if kind == "index":
                qdb = Questdb()
                price_opens = qdb.get_data(
                    f"select date,num,close from minute_data_{kind} where code='{code}'"
                )
                price_opens = price_opens[price_opens.num == "1"]
                price_opens = price_opens.set_index("date").close
                price_opens.index = pd.to_datetime(price_opens.index, format="%Y%m%d")
            else:
                price_opens = read_daily(open=1)[code]
        price_opens = price_opens[price_opens.index >= fac.index.min()]
        self.price_opens = price_opens
        self.price_rets = price_opens.pct_change()
        self.fac = fac
        self.fac_rets = (self.fac.shift(2) * self.price_rets).dropna()
        self.ab_rets = (self.fac_rets - self.price_rets).dropna()
        self.price_rets = self.price_rets.dropna()
        self.fac_nets = (1 + self.fac_rets).cumprod()
        self.ab_nets = (1 + self.ab_rets).cumprod()
        self.price_nets = (1 + self.price_rets).cumprod()
        self.fac_nets = self.fac_nets / self.fac_nets.iloc[0]
        self.ab_nets = self.ab_nets / self.ab_nets.iloc[0]
        self.price_nets = self.price_nets / self.price_nets.iloc[0]
        self.fac_comments = comments_on_twins(self.fac_nets, self.fac_rets)
        self.ab_comments = comments_on_twins(self.ab_nets, self.ab_rets)
        self.price_comments = comments_on_twins(self.price_nets, self.price_rets)
        self.total_comments = pd.concat(
            [self.fac_comments, self.ab_comments, self.price_comments], axis=1
        )
        self.total_nets = pd.concat(
            [self.fac_nets, self.ab_nets, self.price_nets], axis=1
        )
        self.total_rets = pd.concat(
            [self.fac_rets, self.ab_rets, self.price_rets], axis=1
        )
        self.total_comments.columns = (
            self.total_nets.columns
        ) = self.total_rets.columns = ["因子绝对", "因子超额", "买入持有"]
        self.total_comments = np.around(self.total_comments, 3)
        self.iplot = iplot
        self.plot()
        if comments_writer is None and sheetname is not None:
            from pure_ocean_breeze.legacy_version.v3p4.state.states import COMMENTS_WRITER

            comments_writer = COMMENTS_WRITER
            self.total_comments.to_excel(comments_writer, sheet_name=sheetname)
        if net_values_writer is None and sheetname is not None:
            from pure_ocean_breeze.legacy_version.v3p4.state.states import NET_VALUES_WRITER

            net_values_writer = NET_VALUES_WRITER
            self.total_nets.to_excel(net_values_writer, sheet_name=sheetname)

    def plot(self):
        coms = self.total_comments.copy().reset_index()
        if self.iplot:
            figs = cf.figures(
                self.total_nets,
                [dict(kind="line", y=list(self.total_nets.columns))],
                asList=True,
            )
            coms = coms.rename(columns={list(coms)[0]: "绩效指标"})
            table = FF.create_table(coms.iloc[::-1])
            table.update_yaxes(matches=None)
            figs.append(table)
            figs = [figs[-1]] + figs[:-1]
            figs[1].update_layout(
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            base_layout = cf.tools.get_base_layout(figs)
            sp = cf.subplots(
                figs,
                shape=(2, 10),
                base_layout=base_layout,
                vertical_spacing=0.15,
                horizontal_spacing=0.03,
                shared_yaxes=False,
                specs=[
                    [
                        None,
                        {"rowspan": 2, "colspan": 3},
                        None,
                        None,
                        {"rowspan": 2, "colspan": 6},
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                    [
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                ],
            )
            sp["layout"].update(showlegend=True)
            cf.iplot(sp)
        else:
            self.total_nets.plot()
            plt.show()
            tb = Texttable()
            tb.set_cols_width([8] + [7] + [8] * 2 + [7] * 2 + [8])
            tb.set_cols_dtype(["f"] * 7)
            tb.header(list(coms.T.reset_index().columns))
            tb.add_rows(coms.T.reset_index().to_numpy(), header=True)
            print(tb.draw())
