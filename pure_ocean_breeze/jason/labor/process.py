__updated__ = "2025-06-20 03:04:41"

import datetime
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tqdm.auto
import json
import scipy.stats as ss

import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

from functools import reduce, lru_cache
from loguru import logger
from plotly.tools import FigureFactory as FF
import plotly.graph_objects as go

import cufflinks as cf

try:
    cf.set_config_file(offline=True)
except Exception:
    pass
from pure_ocean_breeze.jason.data.read_data import (
    read_daily,
    read_market,
)
from pure_ocean_breeze.jason.state.homeplace import HomePlace

try:
    homeplace = HomePlace()
except Exception:
    print("您暂未初始化，功能将受限")
from pure_ocean_breeze.jason.state.states import STATES
from pure_ocean_breeze.jason.state.decorators import do_on_dfs
from pure_ocean_breeze.jason.data.tools import (
    to_percent,
    standardlize,
    boom_one,
    de_cross_special_for_barra_weekly_fast,
    get_abs,
)
import altair as alt
from IPython.display import display
from IPython.display import display, Markdown
import os
import rust_pyfunc as rp
import time

@lru_cache(maxsize=1)
def get_barras():
    barras={}
    barras['beta']=boom_one(pd.read_parquet(homeplace.barra_data_file+'beta.parquet'),10)
    barras['book_to_price']=boom_one(pd.read_parquet(homeplace.barra_data_file+'booktoprice.parquet'),10)
    barras['earnings_yield']=boom_one(pd.read_parquet(homeplace.barra_data_file+'earningsyield.parquet'),10)
    barras['growth']=boom_one(pd.read_parquet(homeplace.barra_data_file+'growth.parquet'),10)
    barras['leverage']=boom_one(pd.read_parquet(homeplace.barra_data_file+'leverage.parquet'),10)
    barras['liquidity']=boom_one(pd.read_parquet(homeplace.barra_data_file+'liquidity.parquet'),10)
    barras['momentum']=boom_one(pd.read_parquet(homeplace.barra_data_file+'momentum.parquet'),10)
    barras['non_linear_size']=boom_one(pd.read_parquet(homeplace.barra_data_file+'nonlinearsize.parquet'),10)
    barras['residual_volatility']=boom_one(pd.read_parquet(homeplace.barra_data_file+'residualvolatility.parquet'),10)
    barras['size']=boom_one(pd.read_parquet(homeplace.barra_data_file+'size.parquet'),10)
    barras['stock_return']=boom_one(pd.read_parquet(homeplace.barra_data_file+'stockreturn.parquet'),10)
    return barras


@do_on_dfs
def add_cross_standardlize(*args: list) -> pd.DataFrame:
    """将众多因子横截面做z-score标准化之后相加

    Returns
    -------
    `pd.DataFrame`
        合成后的因子
    """
    res = reduce(lambda x, y: x + y, [standardlize(i) for i in args])
    return res


def show_corr(
    fac1: pd.DataFrame,
    fac2: pd.DataFrame,
    method: str = "pearson",
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
        计算相关系数的方法, by default "pearson"
    plt_plot : bool, optional
        是否画出相关系数的时序变化图, by default 1
    show_series : bool, optional
        返回相关性的序列，而非均值
    old_way : bool, optional
        使用3.x版本的方式求相关系数

    Returns
    -------
    `float`
        平均截面相关系数
    """
    corr = fac1.corrwith(fac2, axis=1, method=method)
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
    method: str = "pearson",
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
        计算相关系数的方法, by default "pearson"

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
    corrs=pd.DataFrame(corrs.fillna(0).to_numpy()+corrs.fillna(0).to_numpy().T-np.diag(np.diag(corrs)),index=corrs.index,columns=corrs.columns)
    if show_percent:
        pcorrs = corrs.applymap(to_percent)
    else:
        pcorrs = corrs.copy()
    if print_bool:
        return pcorrs
    else:
        return corrs



class frequency_controller(object):
    def __init__(self, freq: str):
        self.homeplace = HomePlace()
        self.freq = freq
        self.counts_one_year = 50
        self.time_shift = pd.DateOffset(weeks=1)
        self.comment_name = "周"
        self.days_in = 5



class pure_moon(object):
    __slots__ = [
        "homeplace",
        "factors",
        "tradedays",
        "ages",
        "amounts",
        "closes",
        "opens",
        "states",
        "opens_monthly",
        "closes_monthly",
        "rets_monthly",
        "limit_ups",
        "limit_downs",
        "data",
        "ic_icir_and_rank",
        "big_small_rankic",
        "rets_monthly_limit_downs",
        "group_rets",
        "long_short_rets",
        "long_short_net_values",
        "group_net_values",
        "long_short_ret_yearly",
        "long_short_vol_yearly",
        "long_short_info_ratio",
        "long_short_comments",
        "total_comments",
        "__factors_out",
        "ics",
        "rankics",
        "rets_monthly_last",
        "freq_ctrl",
        "freq",
        "factor_cover",
        "factor_cross_skew",
        "pos_neg_rate",
        "corr_itself",
        "rets_all",
        "inner_long_ret_yearly",
        "inner_short_ret_yearly",
        "inner_long_net_values",
        "inner_short_net_values",
        "group_mean_rets_monthly",
        "not_ups",
        "not_downs",
        "group1_ret_yearly",
        "group10_ret_yearly",
        "market_ret",
        "long_minus_market_rets",
        "long_minus_market_nets",
        "inner_rets_long",
        "inner_rets_short",
        "big_rankics",
        "small_rankics",
        'longside_ret_eachyear',
        'longside_ret',
        'alt_name',
        'alt_chart',
    ]

    @classmethod
    @lru_cache(maxsize=None)
    def __init__(
        cls,
        freq: str = "W",
    ):
        cls.homeplace = HomePlace()
        cls.freq = freq
        cls.freq_ctrl = frequency_controller(freq)

    @property
    def factors_out(self):
        return self.__factors_out

    def __call__(self):
        """调用对象则返回因子值"""
        return self.factors_out

    @classmethod
    @lru_cache(maxsize=None)
    def set_basic_data(
        cls,
    ):
        states = read_daily(state=1, start=STATES["START"])
        opens = read_daily(vwap=1, start=STATES["START"])
        closes = read_daily(vwap=1, start=STATES["START"])
        market=read_market(zz500=1,every_stock=0,close=1).resample(cls.freq).last()
        cls.market_ret=market/market.shift(1)-1
        # 交易状态文件
        cls.states = states
        # Monday vwap
        cls.opens = opens
        # Friday vwap
        cls.closes = closes
        cls.opens = cls.opens.replace(0, np.nan)
        cls.closes = cls.closes.replace(0, np.nan)
        cls.states = read_daily(state=1)
        cls.states = cls.states.resample(cls.freq).first()
        up_downs = read_daily(up_down_limit_status=1)
        cls.not_ups = (
            np.sign(up_downs.where(up_downs != 1, np.nan).abs() + 1)
            .resample(cls.freq)
            .first()
        )
        cls.not_downs = (
            np.sign(up_downs.where(up_downs != -1, np.nan).abs() + 1)
            .resample(cls.freq)
            .last()
        )

    def set_factor_df_date_as_index(self, df: pd.DataFrame):
        """设置因子数据的dataframe，因子表列名应为股票代码，索引应为时间"""
        # week_here
        self.factors = df.resample(self.freq).last().dropna(how="all")
        self.factors = (self.factors * self.states).dropna(how="all")
        self.factor_cover = self.factors.count().sum()
        total = self.opens.resample(self.freq).last().reindex(self.factors.index).count().sum()
        self.factor_cover = min(self.factor_cover / total, 1)
        self.factor_cross_skew = self.factors.skew(axis=1).mean()
        pos_num = ((self.factors > 0) + 0).sum().sum()
        neg_num = ((self.factors < 0) + 0).sum().sum()
        self.pos_neg_rate = pos_num / (neg_num + pos_num)
        self.corr_itself = show_corr(self.factors, self.factors.shift(1),plt_plot=0)
        

    @classmethod
    @lru_cache(maxsize=None)
    def get_rets_month(cls):
        """计算每月的收益率，并根据每月做出交易状态，做出删减"""
        # week_here
        cls.opens_monthly = cls.opens.resample(cls.freq).first()
        # week_here
        cls.closes_monthly = cls.closes.resample(cls.freq).last()
        cls.rets_monthly = (cls.closes_monthly - cls.opens_monthly) / cls.opens_monthly
        cls.rets_monthly = cls.rets_monthly.stack().reset_index()
        cls.rets_monthly.columns = ["date", "code", "ret"]


    def get_ic_rankic(cls, df):
        """计算IC和RankIC"""
        df1 = df[["ret", "fac"]]
        rankic = rp.rank_axis0_df(df1).corr().iloc[0, 1]
        small_ic=rp.rank_axis0_df(df1[df1.fac<=df1.fac.median()]).corr().iloc[0, 1]
        big_ic=rp.rank_axis0_df(df1[df1.fac>=df1.fac.median()]).corr().iloc[0, 1]
        df2 = pd.DataFrame({"rankic": [rankic],"small_rankic":[small_ic],"big_rankic":[big_ic]})
        return df2

    def get_icir_rankicir(cls, df):
        """计算ICIR和RankICIR"""
        rankic = df.rankic.mean()
        small_rankic=df.small_rankic.mean()
        big_rankic=df.big_rankic.mean()
        return pd.DataFrame(
            {"IC": [rankic]},
            index=["评价指标"],
        ),pd.DataFrame({"1-5IC":[small_rankic],"6-10IC":[big_rankic]},index=["评价指标"]).T

    def get_ic_icir_and_rank(cls, df):
        """计算IC、ICIR、RankIC、RankICIR"""
        df1 = df.groupby("date").apply(cls.get_ic_rankic)
        cls.rankics = df1.rankic
        cls.rankics = cls.rankics.reset_index(drop=True, level=1).to_frame()
        cls.small_rankics = df1.small_rankic.reset_index(drop=True, level=1).to_frame()
        cls.big_rankics = df1.big_rankic.reset_index(drop=True, level=1).to_frame()
        df2,df5 = cls.get_icir_rankicir(df1)
        df2 = df2.T
        return df2,df5

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

    def get_data(self, groups_num):
        """拼接因子数据和每月收益率数据，并对涨停和跌停股加以处理"""
        self.data = pd.merge(
            self.rets_monthly, self.factors, how="inner", on=["date", "code"]
        )
        self.ic_icir_and_rank,self.big_small_rankic = self.get_ic_icir_and_rank(self.data)
        self.data = self.data.groupby("date").apply(
            lambda x: self.get_groups(x, groups_num)
        )
        self.data = self.data.reset_index(drop=True)

    def to_group_ret(self, l):
        """每一组的年化收益率"""
        # week_here
        ret = l[-1] / len(l) * self.freq_ctrl.counts_one_year
        return ret

    def make_start_to_one(self, l):
        """让净值序列的第一个数变成1"""
        min_date = self.factors.date.min()
        add_date = min_date - pd.DateOffset(weeks=1)
        add_l = pd.Series([0], index=[add_date])
        l = pd.concat([add_l, l])
        return l

    def get_group_rets_net_values(self, groups_num=10):
        """计算组内每一期的平均收益，生成每日收益率序列和净值序列"""
        
        self.group_rets = self.data.groupby(["date", "group"]).apply(
            lambda x: x.ret.mean()
        )
        self.rets_all = self.data.groupby(["date"]).apply(lambda x: x.ret.mean())
        # dropna是因为如果股票行情数据比因子数据的截止日期晚，而最后一个月发生月初跌停时，会造成最后某组多出一个月的数据
        self.group_rets = self.group_rets.unstack()
        self.group_rets = self.group_rets[
            self.group_rets.index <= self.factors.date.max()
        ]
        self.group_rets.columns = list(map(str, list(self.group_rets.columns)))
        self.group_rets = self.group_rets.add_prefix("group")
        self.rets_all=self.rets_all.dropna()
        self.long_short_rets = (
            self.group_rets["group1"] - self.group_rets["group" + str(groups_num)]
        )
        self.inner_rets_long = self.group_rets.group1 - self.rets_all
        self.inner_rets_short = (
            self.rets_all - self.group_rets["group" + str(groups_num)]
        )
        self.long_short_net_values = self.make_start_to_one(
            self.long_short_rets.cumsum()
        )
        self.long_minus_market_rets=self.group_rets.group1-self.market_ret
        
        if self.long_short_net_values[-1] <= self.long_short_net_values[0]:
            self.long_short_rets = (
                self.group_rets["group" + str(groups_num)] - self.group_rets["group1"]
            )
            self.long_short_net_values = self.make_start_to_one(
                self.long_short_rets.cumsum()
            )
            self.inner_rets_long = (
                self.group_rets["group" + str(groups_num)] - self.rets_all
            )
            self.inner_rets_short = self.rets_all - self.group_rets.group1
            self.long_minus_market_rets=self.group_rets['group'+str(groups_num)]-self.market_ret
        self.long_minus_market_nets=self.make_start_to_one(self.long_minus_market_rets.dropna().cumsum())
        self.inner_long_net_values = self.make_start_to_one(
            self.inner_rets_long.cumsum()
        )
        self.inner_short_net_values = self.make_start_to_one(
            self.inner_rets_short.cumsum()
        )
        self.group_rets = self.group_rets.assign(long_short=self.long_short_rets)
        self.group_net_values = self.group_rets.cumsum()
        self.group_net_values = self.group_net_values.apply(self.make_start_to_one)

    def get_long_short_comments(self):
        """计算多空对冲的相关评价指标
        包括年化收益率、年化波动率、信息比率、月度胜率、最大回撤率"""
        # week_here
        self.long_short_ret_yearly = self.long_short_net_values[-1] * (
            self.freq_ctrl.counts_one_year / len(self.long_short_net_values)
        )
        self.inner_long_ret_yearly = self.inner_long_net_values[-1] * (
            self.freq_ctrl.counts_one_year / len(self.inner_long_net_values)
        )
        self.inner_short_ret_yearly = self.inner_short_net_values[-1] * (
            self.freq_ctrl.counts_one_year / len(self.inner_short_net_values)
        )
        
        # week_here
        self.long_short_vol_yearly = np.std(self.long_short_rets) * (
            self.freq_ctrl.counts_one_year**0.5
        )
        self.long_short_info_ratio = (
            self.long_short_ret_yearly / self.long_short_vol_yearly
        )
        self.long_short_comments = pd.DataFrame(
            {
                "评价指标": [
                    self.long_short_ret_yearly,
                    self.long_short_vol_yearly,
                    self.long_short_info_ratio,
                ]
            },
            # week_here
            index=[
                "收益率",
                "波动率",
                "信息比",
            ],
        )

    def get_total_comments(self):
        """综合IC、ICIR、RankIC、RankICIR,年化收益率、年化波动率、信息比率、胜率、最大回撤率"""
        self.group_mean_rets_monthly = self.group_rets.drop(
            columns=["long_short"]
        ).mean()
        mar=self.market_ret.reindex(self.factors_out.index)
        self.group_mean_rets_monthly = (
            self.group_mean_rets_monthly - mar.mean()
        )*self.freq_ctrl.counts_one_year
        self.group1_ret_yearly= self.group_mean_rets_monthly.loc['group1']
        self.group10_ret_yearly = self.group_mean_rets_monthly.loc['group10']
        if self.group1_ret_yearly>self.group10_ret_yearly:
            self.longside_ret=self.group_rets.group1-mar
        else:
            self.longside_ret=self.group_rets.group10-mar
        self.longside_ret_eachyear=self.longside_ret.resample('Y').mean()*self.freq_ctrl.counts_one_year
        self.total_comments = pd.concat(
            [
                self.ic_icir_and_rank,
                self.long_short_comments,
                # week_here
                pd.DataFrame(
                    {
                        "评价指标": [
                            self.pos_neg_rate,
                            self.factor_cross_skew,
                            self.corr_itself,
                            self.factor_cover,
                        ]
                    },
                    index=[
                        "正值占比",
                        "截面偏度",
                        "自相关性",
                        "覆盖率",
                    ],
                ),
                self.big_small_rankic,
                pd.DataFrame(
                    {
                        "评价指标": [
                            self.group1_ret_yearly,
                            self.group10_ret_yearly,
                        ]
                    },
                    index=[
                        "1组收益",
                        "10组收益",
                    ]
                )
            ]
        )
        

    def plot_net_values(self, ilegend=1, without_breakpoint=0):

        tris = self.group_net_values.drop(columns=['long_short'])
        if without_breakpoint:
            tris = tris.dropna()
        figs = cf.figures(
            tris,
            [
                dict(kind="line", y=list(tris.columns)),
            ],
            asList=True,
        )
        comments = (
            self.total_comments.applymap(lambda x: round(x, 4))
            .reset_index()
        )
        here = pd.concat(
            [
                comments.iloc[1:7, :].reset_index(drop=True),
                comments.iloc[[7,0,8,9,10,11], :].reset_index(drop=True),
            ],
            axis=1,
        )
        here.columns = ["绩效", "结果", "多与空", "结果"]
        # here=here.to_numpy().tolist()+[['信息系数','结果','绩效指标','结果']]
        table = FF.create_table(here.iloc[::-1],xgap=0)
        table.update_yaxes(matches=None)
        pic2 = go.Figure(
            go.Bar(
                y=list(self.group_mean_rets_monthly),
                x=[
                    i.replace("roup", "")
                    for i in list(self.group_mean_rets_monthly.index)
                ],
                name='各组收益'
            )
        )
        # table=go.Figure([go.Table(header=dict(values=list(here.columns)),cells=dict(values=here.to_numpy().tolist()))])
        if self.group1_ret_yearly>self.group10_ret_yearly:
            pic3_data = go.Bar(y=list(self.small_rankics.small_rankic), x=list(self.small_rankics.index),marker_color="red")
            pic3 = go.Figure(data=[pic3_data])
            pic5_data1 = go.Scatter(
                y=list(self.small_rankics.small_rankic.cumsum()),
                x=list(self.small_rankics.index),
                name="多头ic",
                yaxis="y2",
                mode="lines",
                line=dict(color="blue"),
            )
        else:
            pic3_data = go.Bar(y=list(self.big_rankics.big_rankic), x=list(self.big_rankics.index),marker_color="red")
            pic3 = go.Figure(data=[pic3_data])
            pic5_data1 = go.Scatter(
                y=list(self.big_rankics.big_rankic.cumsum()),
                x=list(self.big_rankics.index),
                name="多头ic",
                yaxis="y2",
                mode="lines",
                line=dict(color="blue"),
            )
        pic5_data2 = go.Scatter(
            y=list(self.rankics.rankic.cumsum()),
            x=list(self.rankics.index),
            mode="lines",
            name="rankic",
            yaxis="y2",
            line=dict(color="red"),
        )
        figs.append(table)
        figs = [figs[-1]] + figs[:-1]
        figs.append(pic2)
        figs = [figs[0], figs[1], figs[-1], go.Figure()]
        figs[3].add_trace(pic5_data1)
        figs[3].add_trace(pic5_data2)

        pic4=go.Figure()
        figs.append(pic4)
        figs[4].add_trace(go.Bar(x=[str(i) for i in self.longside_ret_eachyear.index.year],y=self.longside_ret_eachyear,name='各年收益'))            
        figs[1].update_layout(
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        ) 
        
        base_layout = cf.tools.get_base_layout(figs)

        sp = cf.subplots(
            figs,
            shape=(3, 10),
            base_layout=base_layout,
            vertical_spacing=0.15,
            horizontal_spacing=0.045,
            shared_yaxes=False,
            specs=[
                [
                    {"rowspan": 3, "colspan": 3},
                    None,
                    None,
                    {"rowspan": 3, "colspan": 3},
                    None,
                    None,
                    {"colspan": 3},
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
                    {"colspan": 3},
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
                    {"colspan": 3},
                    None,
                    None,
                    None,
                ],
            ],
            subplot_titles=[
                "净值曲线",
                "各组月均超均收益",
                "Rank IC时序图",
            ],
        )
        sp["layout"].update(showlegend=ilegend,width=1100,height=200,margin=dict(l=0, r=0, b=0, t=0, pad=0),font=dict(size=12),legend=dict(
            # 可选值：
            # 'left', 'center', 'right' 
            # 'top', 'middle', 'bottom'
            xanchor="right",    # 图例的x锚点
            yanchor="top",     # 图例的y锚点
            x=1,              # x位置（0到1之间）
            y=1               # y位置（0到1之间）
        ))
        cf.iplot(sp)
        
    def check_chart_size(self, chart, unit='KB'):
        """
        检查Altair图表占用的存储空间
        
        Parameters
        ----------
        chart : alt.Chart
            要检查大小的Altair图表对象
        unit : str, optional
            显示单位，可选 'B', 'KB', 'MB', by default 'KB'
        
        Returns
        -------
        float
            图表大小（以指定单位表示）
        """
        import json
        
        # 将图表转换为JSON规范
        chart_json = json.dumps(chart.to_dict())
        size_bytes = len(chart_json.encode('utf-8'))
        
        if unit.upper() == 'B':
            return size_bytes
        elif unit.upper() == 'KB':
            return size_bytes / 1024
        elif unit.upper() == 'MB':
            return size_bytes / (1024 * 1024)
        else:
            raise ValueError("单位必须是 'B', 'KB' 或 'MB'")

    def compare_chart_sizes(self, charts, chart_names=None, unit='KB'):
        """
        比较多个Altair图表的存储空间大小
        
        Parameters
        ----------
        charts : list
            要比较的Altair图表对象列表
        chart_names : list, optional
            图表名称列表，by default None
        unit : str, optional
            显示单位，可选 'B', 'KB', 'MB', by default 'KB'
        
        Returns
        -------
        pd.DataFrame
            包含每个图表大小的数据框
        """
        import pandas as pd
        
        if chart_names is None:
            chart_names = [f"图表{i+1}" for i in range(len(charts))]
        
        sizes = [self.check_chart_size(chart, unit) for chart in charts]
        
        result = pd.DataFrame({
            "图表名称": chart_names,
            f"大小({unit})": sizes
        })
        
        return result.sort_values(f"大小({unit})", ascending=False)

    def plot_net_values_altair(self, ilegend=1, without_breakpoint=0, return_size=False, alt_name='test'):
        """使用Altair库实现相同的可视化效果，布局与原Plotly版本相似，美化版本"""
        # import altair as alt
        
        # 禁用最大行限制，避免大数据集的问题
        alt.data_transformers.disable_max_rows()
        
        # 使用默认的数据转换器以确保兼容性
        alt.data_transformers.enable('default')
        
        # 设置全局宽度和颜色方案
        chart_width = 240  # 增加净值曲线宽度以适应右侧图例
        bar_width = 220  # 增加柱状图宽度
        ic_width = 220     # 增加IC图宽度
        table_width = 240  # 增加表格宽度
        
        # 现代化色彩方案 - 使用更专业的配色
        color_scheme = ['#2E86C1', '#E74C3C', '#28B463', '#F39C12', '#8E44AD', 
                        '#17A589', '#D35400', '#5D6D7E', '#F1C40F', '#E67E22']
        
        # 准备净值曲线数据（包含多空净值）
        tris = self.group_net_values.copy()
        if without_breakpoint:
            tris = tris.dropna()
            
        # 重置索引并获取日期列名
        tris_reset = tris.reset_index()
        date_col = tris_reset.columns[0]
        
        # 分组净值数据（排除多空）
        group_cols = [c for c in tris.columns if c != 'long_short']
        # 减少分组如果min_size为True
        keep_groups = ['group1', 'group5', 'group10'] if 'group10' in group_cols else [group_cols[0], group_cols[-1]]
        keep_groups = [g for g in keep_groups if g in group_cols]
        group_cols = keep_groups
            
        group_data = pd.melt(tris_reset, id_vars=date_col, value_vars=group_cols, var_name='分组', value_name='净值')
        group_data = group_data.rename(columns={date_col: 'date'})
        group_data['净值'] = group_data['净值']
        
        # 多空净值数据（去除缺失以保证连续）
        ls_data = tris_reset[[date_col, 'long_short']].rename(columns={date_col: 'date', 'long_short': '多空净值'}).dropna(subset=['多空净值'])
        ls_data['多空净值'] = ls_data['多空净值']
        
        # 分组净值曲线 - 现代化样式，图例放右侧
        net_group_chart = alt.Chart(group_data).mark_line(
            strokeWidth=2.5,
            opacity=0.8
        ).encode(
            x=alt.X('date:T', axis=alt.Axis(
                labelAngle=-45, 
                labelFontSize=9,
                title=None, 
                grid=False,
                tickCount=6,
                domainColor='#E5E7E9',
                tickColor='#E5E7E9'
            )),
            y=alt.Y('净值:Q', title='净值', axis=alt.Axis(
                labelFontSize=9,
                titleFontSize=11,
                titleColor='#2C3E50',
                titleFontWeight='bold',
                grid=True,
                gridColor='#F8F9FA',
                gridOpacity=0.7,
                tickCount=6,
                domainColor='#E5E7E9'
            )),
            color=alt.Color('分组:N', 
                          scale=alt.Scale(range=color_scheme[:len(group_cols)]), 
                          legend=alt.Legend(
                              orient='right',  # 改为右侧
                              titleFontSize=11,
                              labelFontSize=10,
                              symbolSize=100,
                              padding=15,
                              offset=10,
                              titleAnchor='start'
                          )),
        )
        
        # 多空净值曲线（第二 Y 轴）- 现代化样式
        net_ls_chart = alt.Chart(ls_data).mark_line(
            color='#34495E', 
            strokeWidth=3,
            opacity=0.9
        ).encode(
            x='date:T',
            y=alt.Y('多空净值:Q', title='多空净值', axis=alt.Axis(
                titleFontSize=11,
                titleColor='#2C3E50',
                titleFontWeight='bold',
                labelFontSize=9,
                orient='right',
                tickCount=6,
                domainColor='#E5E7E9'
            ))
        )
        
        # 合并双轴图层
        net_value_chart = alt.layer(net_group_chart, net_ls_chart).resolve_scale(y='independent').properties(
            width=chart_width,
            height=120,
            title=alt.TitleParams('净值曲线', fontSize=13, anchor='middle', color='#2C3E50', fontWeight='bold')
        )
        
        # 准备表格数据
        comments = self.total_comments.applymap(lambda x: round(x, 3)).reset_index()
        
        # 创建转置表格数据
        table_data_orig = pd.concat(
            [
                comments.iloc[1:7, :].reset_index(drop=True),
                comments.iloc[[7,0,8,9,10,11], :].reset_index(drop=True),
            ],
            axis=1,
        )
        table_data_orig.columns = ["绩效", "结果1", "多与空", "结果2"]
        
        # 转置表格数据
        table_data = pd.DataFrame({
            "指标": ["收益率", "波动率", "信息比", "正值占比", "截面偏度", "自相关性"],
            "结果": table_data_orig["结果1"].values,
            "多空指标": ["覆盖率", "IC", "1-5IC", "6-10IC", "1组收益", "10组收益"],
            "多空结果": table_data_orig["结果2"].values
        })
        
        # 创建梦幻美观的表格
        def create_dreamy_table(data):
            """创建梦幻美观的表格，使用渐变和现代化样式"""
            df = data.copy().reset_index(drop=True)
            rows = []
            for r in range(len(df)):
                for c, col in enumerate(df.columns):
                    val = df.iloc[r, c]
                    text = f"{val:.3f}" if isinstance(val, float) else str(val)
                    rows.append({'row': str(r), 'col': col, 'text': text})
            
            # 不需要表头行
            plot_df = pd.DataFrame(rows)
            
            # 定义行顺序：只包含数据行
            row_order = [str(r) for r in range(len(df))]
            
            # 基础图层
            base = alt.Chart(plot_df).encode(
                x=alt.X('col:O', axis=None, sort=list(data.columns)),
                y=alt.Y('row:O', axis=None, sort=row_order)
            ).properties(width=table_width, height=120)
            
            # 数据行背景 - 使用梦幻色彩交替
            data_bg_even = base.mark_rect(
                fill='#EBF5FB',  # 浅蓝色
                stroke='#D6EAF8', 
                strokeWidth=1,
                opacity=0.8
            ).transform_filter(
                alt.expr.parseInt(alt.datum.row) % 2 == 0
            )
            
            data_bg_odd = base.mark_rect(
                fill='#F4ECF7',  # 浅紫色
                stroke='#E8DAEF', 
                strokeWidth=1,
                opacity=0.8
            ).transform_filter(
                alt.expr.parseInt(alt.datum.row) % 2 == 1
            )
            
            # 单元格文字 - 使用深色，增强对比度
            cell_text = base.mark_text(
                baseline='middle', 
                align='center', 
                fontSize=12,  # 增大字体从10到12
                color='#1B2631',  # 深色文字
                fontWeight=600  # 增加字重
            ).encode(
                text='text:N'
            )
            
            # 添加表格边框装饰 - 使用连贯直线
            border_decoration = base.mark_rect(
                fill='transparent',
                stroke='white',  # 改为白色边框
                strokeWidth=2,
                opacity=0.6
            )
            
            return data_bg_even + data_bg_odd + border_decoration + cell_text
        
        # 创建表格
        table_chart = create_dreamy_table(table_data).properties(
            title=alt.TitleParams('评价指标', fontSize=13, anchor='middle', color='#2C3E50', fontWeight='bold')
        )
        
        # 准备柱状图数据并应用范围限制
        bar_data = pd.DataFrame({
            '分组': [j.replace('g','g0') if len(j)==2 else j for j in [i.replace("roup", "") for i in list(self.group_mean_rets_monthly.index)]],
            '收益率': list(self.group_mean_rets_monthly)
        })
        
        # 对收益率数据进行范围限制：正值0-0.15，负值0到-0.25
        bar_data['收益率_限制'] = bar_data['收益率'].clip(-0.25, 0.15)
        
        # 各组收益柱状图 - 现代化样式，固定y轴范围
        bar_chart = alt.Chart(bar_data).mark_bar(
            cornerRadius=3,
            stroke='white',
            strokeWidth=1
        ).encode(
            x=alt.X('分组:N', title=None, axis=alt.Axis(
                labelAngle=0,
                labelFontSize=9,
                domainColor='#E5E7E9',
                tickColor='#E5E7E9'
            )),
            y=alt.Y('收益率_限制:Q', 
                   title='收益率', 
                   scale=alt.Scale(domain=[-0.25, 0.15]),  # 固定y轴范围
                   axis=alt.Axis(
                       labelFontSize=9,
                       titleFontSize=11,
                       titleColor='#2C3E50',
                       titleFontWeight='bold',
                       grid=True,
                       gridColor='#F8F9FA',
                       gridOpacity=0.7,
                       tickCount=8,
                       domainColor='#E5E7E9'
                   )),
            color=alt.condition(
                alt.datum.收益率_限制 > 0,
                alt.value('#E74C3C'),  # 正值用红色
                alt.value('#28B463')   # 负值用绿色
            ),
            opacity=alt.value(0.8)
        ).properties(
            width=bar_width,
            height=120,
            title=alt.TitleParams('各组月均超均收益', fontSize=13, anchor='middle', color='#2C3E50', fontWeight='bold')
        )
        
        # 准备IC数据并应用范围限制
        if self.group1_ret_yearly > self.group10_ret_yearly:
            ic_data = self.small_rankics.small_rankic.reset_index().dropna()  # 移除nan值
            ic_data.columns = ['date', 'ic_value']
            ic_label = "多头ic"
        else:
            ic_data = self.big_rankics.big_rankic.reset_index().dropna()  # 移除nan值
            ic_data.columns = ['date', 'ic_value']
            ic_label = "多头ic"
        
        # 对IC数据进行范围限制：-0.3到0.3
        ic_data['ic_value_限制'] = ic_data['ic_value'].clip(-0.3, 0.3)
        
        ic_cum = ic_data.copy()
        ic_cum['ic_cum'] = ic_data['ic_value'].cumsum()
        
        rankic_data = self.rankics.rankic.reset_index().dropna()  # 移除nan值
        rankic_data.columns = ['date', 'rankic_value']
        rankic_cum = rankic_data.copy()
        rankic_cum['rankic_cum'] = rankic_data['rankic_value'].cumsum()
        
        # 创建现代化IC图表 - 柱状图加折线图，固定y轴范围
        base = alt.Chart(ic_data).encode(
            x=alt.X('date:T', axis=alt.Axis(
                labelAngle=-45, 
                labelFontSize=9,
                title=None,
                tickCount=6,
                domainColor='#E5E7E9',
                tickColor='#E5E7E9'
            )),
        )
        
        # IC柱状图层 - 固定y轴范围
        bars = base.mark_bar(
            color='#F39C12', 
            opacity=0.7, 
            size=2,
            stroke='white',
            strokeWidth=0.5
        ).encode(
            y=alt.Y('ic_value_限制:Q', 
                   title='IC值',
                   scale=alt.Scale(domain=[-0.3, 0.3]),  # 固定y轴范围
                   axis=alt.Axis(
                       titleFontSize=11,
                       titleColor='#2C3E50',
                       titleFontWeight='bold',
                       labelFontSize=9,
                       grid=True,
                       gridColor='#F8F9FA',
                       gridOpacity=0.7,
                       tickCount=7,
                       domainColor='#E5E7E9'
                   ))
        )
        
        # 累计IC线图层 - 现代化样式
        line1 = alt.Chart(ic_cum).mark_line(
            color='#2E86C1', 
            strokeWidth=2.5,
            opacity=0.9
        ).encode(
            x='date:T',
            y=alt.Y('ic_cum:Q', 
                    title='累计IC',
                    axis=alt.Axis(
                        titleFontSize=11,
                        titleColor='#2E86C1',
                        titleFontWeight='bold',
                        labelFontSize=9,
                        tickCount=6,
                        domainColor='#E5E7E9'
                    ))
        )
        
        # RankIC累计线图层 - 现代化样式
        line2 = alt.Chart(rankic_cum).mark_line(
            color='#E74C3C', 
            strokeWidth=2.5,
            opacity=0.9,
        ).encode(
            x='date:T',
            y=alt.Y('rankic_cum:Q')
        )
        
        # 将柱状图和线图图层结合
        ic_chart = alt.layer(
            bars,
            line1.encode(y=alt.Y('ic_cum:Q', axis=alt.Axis(title='累计IC', titleColor='#2E86C1', titleFontWeight='bold'))),
            line2.encode(y=alt.Y('rankic_cum:Q', axis=None))
        ).resolve_scale(
            y='independent'
        ).properties(
            width=ic_width,
            height=120,
            title=alt.TitleParams('Rank IC时序图(蓝色多头)', fontSize=13, anchor='middle', color='#2C3E50', fontWeight='bold')
        )
        
        # 获取当前时间戳
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 创建现代化布局：所有子图横向排布
        combined_chart = alt.hconcat(
            table_chart, net_value_chart, bar_chart, ic_chart,
            spacing=20  # 增加图表间距以适应右侧图例
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            domainWidth=1,
            domainColor='#E5E7E9',
            labelLimit=100
        ).configure_title(
            fontSize=12,
            anchor='middle',
            color='#2C3E50',
            fontWeight='bold'
        ).configure_legend(
            titleFontSize=11,
            labelFontSize=10,
            symbolSize=100,
            padding=15
        ).properties(
            title=alt.TitleParams(f"{alt_name} - {current_time}", fontSize=14, anchor='middle', color='#2C3E50', fontWeight='bold')
        )
        
        # 应用最小化配置
        combined_chart: alt.HConcatChart = combined_chart.configure(
            autosize={'type': 'fit', 'contains': 'padding'},
            background='#FEFEFE',  # 设置背景色
            padding={'left': 20, 'top': 20, 'right': 30, 'bottom': 20}  # 增加右边距以适应图例
        )
        
        self.alt_chart = [table_chart, net_value_chart, bar_chart, ic_chart]
        
        # 如果需要返回图表大小
        if return_size:
            chart_size = self.check_chart_size(combined_chart, 'KB')
            return combined_chart, chart_size
        
        # 返回图表
        return combined_chart

    @classmethod
    @lru_cache(maxsize=None)
    def prerpare(cls):
        """通用数据准备"""
        cls.get_rets_month()

    def run(
        self,
        groups_num=10,
        ilegend=1,
        without_breakpoint=0,
        show_more_than=0.025,
        plot_style="altair", # 新增参数，可选 "plotly", "seaborn", "altair"
        return_size=False,   # 新增参数，返回图表大小
        alt_name=None,
        show_alt_chart=True,
    ):
        """运行回测部分"""
        self.__factors_out = self.factors.copy()
        self.factors = self.factors.shift(1)
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ["date", "code", "fac"]
        self.get_data(groups_num)
        self.get_group_rets_net_values(groups_num=groups_num)
        self.get_long_short_comments()

        if (show_more_than is None) or (show_more_than < max(self.group1_ret_yearly,self.group10_ret_yearly)):
            if plot_style == "plotly":
                # 步骤6: 绘制Plotly图表
                chart = self.plot_net_values(
                    ilegend=bool(ilegend),
                    without_breakpoint=without_breakpoint,
                )
                if return_size:
                    return chart, None  # Plotly图表不提供大小检查
            elif plot_style == "altair":
                if return_size:
                    # 步骤6: 绘制Altair图表(带大小返回)
                    chart, size = self.plot_net_values_altair(
                        ilegend=bool(ilegend),
                        without_breakpoint=without_breakpoint,
                        return_size=True,
                        alt_name=alt_name,
                    )
                    
                    import altair as alt
                    try:
                        # 步骤7: 显示图表
                        from IPython.display import display
                        display(chart)
                        return chart, size
                    except ImportError:
                        # 步骤7: 保存HTML文件
                        chart.save('factor_analysis.html')
                        return chart, size
                else:
                    # 步骤6: 绘制Altair图表
                    chart = self.plot_net_values_altair(
                        ilegend=bool(ilegend),
                        without_breakpoint=without_breakpoint,
                        alt_name=alt_name,
                    )
                    
                    import altair as alt
                    try:
                        if show_alt_chart:
                            # 步骤7: 显示图表
                            display_alt_chart(chart,alt_name)
                        # 返回HTML对象而不是显示它，让调用者决定如何处理
                        # HTML(f'<img src="{ipynb_name}/{file_name}.svg">')
                        
                        return chart
                    except ImportError:
                        # 步骤7: 保存HTML文件
                        chart.save('factor_analysis.html')
                        return chart
        else:
            alt_name_prefix=alt_name.replace('neu','')
            logger.info(f'{alt_name_prefix}多头收益率为{round(max(self.group1_ret_yearly,self.group10_ret_yearly),3)}, ic为{round(self.rankics.rankic.mean(),3)}，表现太差，不展示了')
        


def display_image_as_markdown(image_path: str, alt_text: str = "Image", force_reload: bool = True):
    """
    通过生成并显示 Markdown 链接来展示指定的图片文件。
    这种方法不会增加 .ipynb 文件的体积。

    参数:
    image_path (str): 相对于 .ipynb 文件的图片路径。
                    例如 "./altair_charts/my_chart.svg" 或 "images/photo.png"。
    alt_text (str, optional): 图片的 Alt 文本。默认为 "Image"。
    force_reload (bool, optional): 是否强制浏览器重新加载图像。默认为 True。
    """
    markdown_to_display = ""
    if os.path.exists(image_path):
        # 确保路径分隔符是 / (正斜杠) 以便 Markdown 正确解析
        # 同时，为了安全，确保路径是相对的，并且没有 ".." 来跳出当前工作目录太远
        # (虽然这里的os.path.exists已经部分处理了路径有效性)
        normalized_path = os.path.normpath(image_path) # 标准化路径
        markdown_path = normalized_path.replace(os.sep, '/')

        # 一个简单的安全检查，防止路径过于随意 (可选，但有时有益)
        # if markdown_path.startswith("../"):
        #     print(f"警告: 路径 '{markdown_path}' 可能指向项目外部，请谨慎使用。")

        # 添加时间戳参数以防止浏览器缓存
        if force_reload:
            timestamp = int(time.time())
            markdown_path = f"{markdown_path}?t={timestamp}"

        markdown_to_display = f"![{alt_text}]({markdown_path})"
    else:
        markdown_to_display = f"*图片未找到: {image_path}*"

    display(Markdown(markdown_to_display))
    
def display_alt_chart(chart: alt.HConcatChart,alt_name:str,neu_ret:float):
    # 保存图片并通过HTML引用，避免使用display增加ipynb文件大小
    def get_file_name():
        from IPython import get_ipython

        ip = get_ipython()
        path = None
        if "__vsc_ipynb_file__" in ip.user_ns:
            path = ip.user_ns["__vsc_ipynb_file__"]
        else:
            import urllib.parse
            path=ip.parent_header['metadata']['cellId'].split('/')
            path=[i for i in path if '.ipynb' in i][0].split('.ipynb')[0]
            path=urllib.parse.unquote(path)
        return path.split("/")[-1].split(".")[0]
    # 获取当前ipynb文件路径
    pythoncode_dir = '/home/chenzongwei/pythoncode'
    ipynb_path = pythoncode_dir+'/pngs/'
    ipynb_name = get_file_name()
    # 在/home/chenzongwei/pythoncode中查找以ipynb_name命名的ipynb文件
    found_folder = None
    
    for folder_name in os.listdir(pythoncode_dir):
        folder_path = os.path.join(pythoncode_dir, folder_name)
        if os.path.isdir(folder_path):
            # 查找该文件夹中是否有以ipynb_name命名的ipynb文件
            target_file = f"{ipynb_name}.ipynb"
            target_path = os.path.join(folder_path, target_file)
            if os.path.exists(target_path):
                found_folder = folder_name
                break

    ipynb_path = os.path.join(ipynb_path, found_folder)
    os.makedirs(ipynb_path, exist_ok=True)

    # 创建与ipynb同名的文件夹
    save_dir = os.path.join(ipynb_path, ipynb_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图表为png
    file_name = alt_name if alt_name else 'factor_analysis'
    save_path = os.path.join(save_dir, f"{file_name}.svg")
    chart.save(save_path)

    # 将neu_ret以json格式写入neu_rets.json文件
    neu_rets_file = os.path.join(save_dir, "neu_rets.json")
    
    # 读取现有的json文件内容，如果文件不存在则创建空字典
    try:
        if os.path.exists(neu_rets_file):
            with open(neu_rets_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # 检查文件是否为空
                    neu_rets_data = json.loads(content)
                else:
                    neu_rets_data = {}
        else:
            neu_rets_data = {}
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"读取JSON文件时发生错误: {e}，创建新的数据字典")
        neu_rets_data = {}
    
    # 更新数据
    neu_rets_data[file_name] = neu_ret
    
    # 写入文件
    try:
        with open(neu_rets_file, 'w', encoding='utf-8') as f:
            json.dump(neu_rets_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"写入JSON文件时发生错误: {e}")
    display_image_as_markdown(save_path)

@do_on_dfs
class pure_moonnight(object):
    """封装选股框架"""

    __slots__ = ["shen",'chart','size']

    def __init__(
        self,
        factors: pd.DataFrame,
        groups_num: int = 10,
        freq: str = "W",
        time_start: int = None,
        time_end: int = None,
        ilegend: bool = 1,
        without_breakpoint: bool = 0,
        show_more_than: float = 0.025,
        plot_style: str = "altair",
        return_size: bool = False,  # 新增参数，返回图表大小
        alt_name: str = None,
        show_alt_chart: bool = True,
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
        freq : str, optional
            回测频率, by default 'M'
        time_start : int, optional
            回测起始时间, by default None
        time_end : int, optional
            回测终止时间, by default None
        ilegend : bool, optional
            使用cufflinks绘图时，是否显示图例, by default 1
        without_breakpoint : bool, optional
            画图的时候是否去除间断点, by default 0
        show_more_than : float, optional
            展示收益率大于多少的因子，默认展示大于0.025的因子, by default 0.025
        plot_style : str, optional
            绘图风格，可选 "plotly", "seaborn", "altair", by default "plotly"
        min_size : bool, optional
            是否使用最小化图表模式以减小文件大小, by default False
        return_size : bool, optional
            是否返回图表大小信息, by default False
        """
        if time_start is not None:
            factors = factors[factors.index >= pd.Timestamp(str(time_start))]
        if time_end is not None:
            factors = factors[factors.index <= pd.Timestamp(str(time_end))]
        self.shen = pure_moon(freq=freq)
        self.shen.set_basic_data()
        self.shen.set_factor_df_date_as_index(factors)
        self.shen.prerpare()
        result = self.shen.run(
            groups_num=groups_num,
            ilegend=ilegend,
            without_breakpoint=without_breakpoint,
            show_more_than=show_more_than,
            plot_style=plot_style,
            return_size=return_size,
            alt_name=alt_name,
            show_alt_chart=show_alt_chart,
        )
        # 如果需要返回大小信息
        if return_size:
            self.chart, self.size = result




@do_on_dfs
def get_group(df: pd.DataFrame, group_num: int = 10) -> pd.DataFrame:
    """使用groupby的方法，将一组因子值改为截面上的分组值，此方法相比qcut的方法更加稳健，但速度更慢一些

    Parameters
    ----------
    df : pd.DataFrame
        因子值，index为时间，columns为股票代码，values为因子值
    group_num : int, optional
        分组的数量, by default 10

    Returns
    -------
    pd.DataFrame
        转化为分组值后的df，index为时间，columns为股票代码，values为分组值
    """
    a = pure_moon()
    df = df.stack().reset_index()
    df.columns = ["date", "code", "fac"]
    df = a.get_groups(df, group_num).pivot(index="date", columns="code", values="group")
    return df


def symmetrically_orthogonalize(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """对多个因子做对称正交，每个因子得到正交其他因子后的结果

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        多个要做正交的因子，每个df都是index为时间，columns为股票代码，values为因子值的df

    Returns
    -------
    list[pd.DataFrame]
        对称正交后的各个因子
    """

    def sing(dfs: list[pd.DataFrame], date: pd.Timestamp):
        dds = []
        for num, i in enumerate(dfs):
            i = i[i.index == date]
            i.index = [f"fac{num}"]
            i = i.T
            dds.append(i)
        dds = pd.concat(dds, axis=1)
        cov = dds.cov()
        d, u = np.linalg.eig(cov)
        d = np.diag(d ** (-0.5))
        new_facs = pd.DataFrame(
            np.dot(dds, np.dot(np.dot(u, d), u.T)), columns=dds.columns, index=dds.index
        )
        new_facs = new_facs.stack().reset_index()
        new_facs.columns = ["code", "fac_number", "fac"]
        new_facs = new_facs.assign(date=date)
        dds = []
        for num, i in enumerate(dfs):
            i = new_facs[new_facs.fac_number == f"fac{num}"]
            i = i.pivot(index="date", columns="code", values="fac")
            dds.append(i)
        return dds

    dfs = [standardlize(i) for i in dfs]
    date_first = max([i.index.min() for i in dfs])
    date_last = min([i.index.max() for i in dfs])
    dfs = [i[(i.index >= date_first) & (i.index <= date_last)] for i in dfs]
    fac_num = len(dfs)
    ddss = [[] for i in range(fac_num)]
    for date in tqdm.auto.tqdm(dfs[0].index):
        dds = sing(dfs, date)
        for num, i in enumerate(dds):
            ddss[num].append(i)
    ds = []
    for i in tqdm.auto.tqdm(ddss):
        ds.append(pd.concat(i))
    return ds
    

@do_on_dfs
def sun(factor:pd.DataFrame,rolling_days:int=10,time_start:int=20170101,show_more_than:float=0.025,plot_style:str='altair',alt_name:str='test1'):
    '''先单因子测试，再测试其与常用风格之间的关系'''
    try:
        
        # 步骤1: 因子排序
        # factor=factor.rank(axis=1)
        factor=rp.rank_axis1_df(factor)
        
        # 步骤2: boom_one处理
        ractor=boom_one(factor,rolling_days)
        
        # 步骤3: 中性化处理
        pfi=de_cross_special_for_barra_weekly_fast(ractor.copy())
        
        # 步骤4: 中性化后的回测
        shen=pure_moonnight(pfi,time_start=time_start,show_more_than=show_more_than,plot_style=plot_style,alt_name=alt_name+'_neu',show_alt_chart=False)
        neu_ret=max(shen.shen.group1_ret_yearly,shen.shen.group10_ret_yearly)
        
        if neu_ret > show_more_than:
            chart1=shen.shen.alt_chart
            
            # 步骤5: 原始值回测
            shen=pure_moonnight(ractor,time_start=time_start,show_more_than=None,plot_style=plot_style,alt_name=alt_name+'_raw',show_alt_chart=False)
            chart2=shen.shen.alt_chart
            
            # 步骤6: 计算与Barra因子的相关性
            corrs={}
            barras=get_barras()
            for k,v in barras.items():
                corrs[k]=rp.corrwith(ractor,v,axis=1).mean()
            corrs=pd.DataFrame(corrs,index=['C'])
            
            # 步骤7: 创建相关性热图
            # 创建相关性热图
            corrs_melted = corrs.reset_index().melt(id_vars='index', var_name='风格因子', value_name='相关性')
            # 按照相关性绝对值从大到小排序
            corrs_melted['相关性绝对值'] = corrs_melted['相关性'].abs()
            corrs_melted = corrs_melted.sort_values('相关性绝对值', ascending=False)
            # 创建一个排序顺序列表，按照相关性绝对值排序
            sort_order = corrs_melted['风格因子'].tolist()
            corr_chart = alt.Chart(corrs_melted).mark_bar().encode(
                x=alt.X('风格因子:N', sort=sort_order, axis=alt.Axis(labelAngle=0, labelFontSize=12, labelFontWeight='bold', title=None)),
                y=alt.Y('相关性:Q', axis=alt.Axis(labelFontSize=12, titleFontSize=9)),
                color=alt.Color('相关性:Q', scale=alt.Scale(scheme='blueorange'))
            ).properties(
                # title=alt.TitleParams('因子与风格因子相关性', fontSize=10, anchor='middle'),
                width=1000,
                height=50
            )
            
            # 添加百分比标签
            text_labels = alt.Chart(corrs_melted).mark_text(
                align='center',
                baseline='middle',
                dy=-10,
                fontSize=10
            ).encode(
                x=alt.X('风格因子:N', sort=sort_order),
                y=alt.Y('相关性:Q'),
                text=alt.Text('相关性:Q', format='.0%')  # 以百分比形式显示，只保留整数
            )
            
            # 组合图表和标签
            corr_chart = alt.layer(corr_chart, text_labels)

            # 步骤8: 创建复合图表布局
            # 创建复合图表布局
            # 提取chart1和chart2中的表格、净值图、柱状图和IC图
            table1, netval1, bar1, ic1 = chart1
            # 修改table1的标题为"中性化"
            table1 = table1.properties(title="中性化")
            
            table2, netval2, bar2, ic2 = chart2
            # 修改table2的标题为"原始值"
            table2 = table2.properties(title="原始值")
            
            # 创建两行的布局
            # 第一行：左侧是中性化因子的表格+净值图，右侧是原始因子的表格+净值图
            row1_left = table1 | netval1
            row1_right = bar1 | ic1
            row1 = row1_left | row1_right
            
            # 第二行：左侧是中性化因子的柱状图+IC图，右侧是原始因子的柱状图+IC图
            row2_left = table2 | netval2
            row2_right = bar2 | ic2
            row2 = row2_left | row2_right
            
            # 第三行：相关性图表
            row3 = corr_chart
            
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 将三行组合
            combined_chart = alt.vconcat(row1, row2, row3).resolve_scale(
                color='independent'
            ).properties(
                title=alt.TitleParams(f"{alt_name} - {current_time}", fontSize=16, anchor='middle')
            )
            
            # 步骤9: 显示组合图表
            display_alt_chart(combined_chart, alt_name,neu_ret)
            
            return bool(neu_ret > show_more_than)
        
        return bool(neu_ret > show_more_than)
    except Exception as e:
        # 将异常转换为字符串形式，在Altair图表中显示
        # raise 
        print(e)
        return False


def suns(names:list[str],facs:list[pd.DataFrame],do_pri:int=1,do_abs:int=0,do_fold:int=1,other_keyword:str=None):
    okays={}
    for name,fac in zip(names,facs):
        if (other_keyword is None) or (other_keyword in name):
            if do_pri:
                # logger.info(name)
                judge=sun(fac,alt_name=name)
                if judge:
                    okays[name]=fac
            if do_abs:
                # logger.info(name+'_abs')
                judge=sun(fac.abs(),alt_name=name+'_abs')
                if judge:
                    okays[name+'_abs']=fac.abs()
            if do_fold:
                # logger.info(name+'_fold')
                judge=sun(get_abs(fac),alt_name=name+'_fold')
                if judge:
                    okays[name+'_fold']=get_abs(fac)
    return okays

def suns_pair(names:list[str],facs:list[pd.DataFrame],keyword1:str,keyword2:str,other_keyword:str=None,do_pri:int=1,do_abs:int=0,do_rate:int=0):
    dacs={k:v for k,v in zip(names,facs)}
    okays={}
    for name,fac in zip(names,facs):
        try:
            if (other_keyword is None) or (other_keyword in name):
                if not do_rate:
                    if keyword1 in name:
                        if do_pri:
                            # logger.info(name.replace(keyword1,f'{keyword1}vs{keyword2}'))
                            judge=sun(fac-dacs[name.replace(keyword1,keyword2)],alt_name=name.replace(keyword1,f'{keyword1}vs{keyword2}'))
                            if judge:
                                okays[name.replace(keyword1,f'{keyword1}vs{keyword2}')]=fac-dacs[name.replace(keyword1,keyword2)]
                        if do_abs:
                            # logger.info(name.replace(keyword1,f'{keyword1}vs{keyword2}')+'_abs')
                            judge=sun((fac-dacs[name.replace(keyword1,keyword2)]).abs(),alt_name=name.replace(keyword1,f'{keyword1}vs{keyword2}')+'_abs')
                            if judge:
                                okays[name.replace(keyword1,f'{keyword1}vs{keyword2}')+'_abs']=(fac-dacs[name.replace(keyword1,keyword2)]).abs()
                else:
                    if keyword1 in name:
                        if do_pri:
                            # logger.info(name.replace(keyword1,f'{keyword1}vs{keyword2}')+'_rate')
                            judge=sun((fac-dacs[name.replace(keyword1,keyword2)])/(fac+dacs[name.replace(keyword1,keyword2)]),alt_name=name.replace(keyword1,f'{keyword1}vs{keyword2}')+'_rate')
                            if judge:
                                okays[name.replace(keyword1,f'{keyword1}vs{keyword2}')+'_rate']=(fac-dacs[name.replace(keyword1,keyword2)])/(fac+dacs[name.replace(keyword1,keyword2)])
                        if do_abs:
                            # logger.info(name.replace(keyword1,f'{keyword1}vs{keyword2}')+'_rate_abs')
                            judge=sun((fac-dacs[name.replace(keyword1,keyword2)]).abs()/(fac.abs()+dacs[name.replace(keyword1,keyword2)].abs()),alt_name=name.replace(keyword1,f'{keyword1}vs{keyword2}')+'_rate_abs')
                            if judge:
                                okays[name.replace(keyword1,f'{keyword1}vs{keyword2}')+'_rate_abs']=(fac-dacs[name.replace(keyword1,keyword2)]).abs()/(fac.abs()+dacs[name.replace(keyword1,keyword2)].abs())
        except Exception as e:
            print(e)
    return okays