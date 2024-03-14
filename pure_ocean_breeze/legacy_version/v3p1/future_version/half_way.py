__updated__ = "2022-08-31 21:50:03"

import numpy as np
import pandas as pd
import knockknock as kk
import matplotlib.pyplot as plt

plt.style.use(["science", "no-latex", "notebook"])
plt.rcParams["axes.unicode_minus"] = False
import plotly.express as pe
import plotly.io as pio
from functools import reduce
from pure_ocean_breeze.labor.process import pure_moon, decap_industry
from pure_ocean_breeze.data.read_data import read_daily
from pure_ocean_breeze.state.homeplace import HomePlace


class pure_cloud(object):
    """
    为了测试其他不同的频率而设计的类，仅考虑了上市满60天这一要素
    这一回测采取的方案是，对于回测频率n天，将初始资金等分成n笔，每天以1/n的资金调仓
    每笔资金之间相互独立，最终汇聚成一个收益率序列
    """

    def __init__(
        self,
        fac,
        freq,
        group=10,
        boxcox=1,
        trade_cost=0,
        print_comments=1,
        plt_plot=1,
        plotly_plot=0,
        filename="净值走势图",
        comments_writer=None,
        nets_writer=None,
        sheet_name=None,
    ):
        """n是回测的频率，等分成n份，group是回测的组数，boxcox是是否做行业市值中性化"""
        self.fac = fac
        self.freq = freq
        self.group = group
        self.boxcox = boxcox
        self.trade_cost = trade_cost
        moon = pure_moon()
        moon.prerpare()
        ages = moon.ages.copy()
        ages = (ages >= 60) + 0
        self.ages = ages.replace(0, np.nan)
        self.closes = read_daily(close=1)
        self.rets = (
            (self.closes.shift(-self.freq) / self.closes - 1) * self.ages
        ) / self.freq
        self.run(
            print_comments=print_comments,
            plt_plot=plt_plot,
            plotly_plot=plotly_plot,
            filename=filename,
        )
        if comments_writer:
            if sheet_name:
                self.long_short_comments.to_excel(
                    comments_writer, sheet_name=sheet_name
                )
            else:
                raise AttributeError("必须制定sheet_name参数🤒")
        if nets_writer:
            if sheet_name:
                self.group_nets.to_excel(nets_writer, sheet_name=sheet_name)
            else:
                raise AttributeError("必须制定sheet_name参数🤒")

    def comments(self, series, series1):
        """对twins中的结果给出评价
        评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率"""
        ret = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
        duration = (series.index[-1] - series.index[0]).days
        year = duration / 365
        ret_yearly = (series.iloc[-1] / series.iloc[0]) ** (1 / year) - 1
        max_draw = -(series / series.expanding(1).max() - 1).min()
        vol = np.std(series1) * (250**0.5)
        sharpe = ret_yearly / vol
        wins = series1[series1 > 0]
        win_rate = len(wins) / len(series1)
        return pd.Series(
            [ret, ret_yearly, vol, sharpe, max_draw, win_rate],
            index=["总收益率", "年化收益率", "年化波动率", "信息比率", "最大回撤率", "胜率"],
        )

    @kk.desktop_sender(title="嘿，变频回测结束啦～🗓")
    def run(self, print_comments, plt_plot, plotly_plot, filename):
        """对因子值分组并匹配"""
        if self.boxcox:
            self.fac = decap_industry(self.fac,daily=1)
        self.fac = self.fac.T.apply(
            lambda x: pd.qcut(x, self.group, labels=False, duplicates="drop")
        ).T
        self.fac = self.fac.shift(1)
        self.vs = [
            (((self.fac == i) + 0).replace(0, np.nan) * self.rets).mean(axis=1)
            for i in range(self.group)
        ]
        self.group_rets = pd.DataFrame(
            {f"group{k}": list(v) for k, v in zip(range(1, self.group + 1), self.vs)},
            index=self.vs[0].index,
        )
        self.group_rets = self.group_rets.dropna(how="all")
        self.group_rets = self.group_rets
        self.group_nets = (self.group_rets + 1).cumprod()
        self.group_nets = self.group_nets.apply(lambda x: x / x.iloc[0])
        self.one = self.group_nets["group1"]
        self.end = self.group_nets[f"group{self.group}"]
        if self.one.iloc[-1] > self.end.iloc[-1]:
            self.long_name = "group1"
            self.short_name = f"group{self.group}"
        else:
            self.long_name = f"group{self.group}"
            self.short_name = "group1"
        self.long_short_ret = (
            self.group_rets[self.long_name] - self.group_rets[self.short_name]
        )
        self.long_short_net = (self.long_short_ret + 1).cumprod()
        self.long_short_net = self.long_short_net / self.long_short_net.iloc[0]
        if self.long_short_net.iloc[-1] < 1:
            self.long_short_ret = (
                self.group_rets[self.short_name] - self.group_rets[self.long_name]
            )
            self.long_short_net = (self.long_short_ret + 1).cumprod()
            self.long_short_net = self.long_short_net / self.long_short_net.iloc[0]
            self.long_short_ret = (
                self.group_rets[self.short_name]
                - self.group_rets[self.long_name]
                - 2 * self.trade_cost / self.freq
            )
            self.long_short_net = (self.long_short_ret + 1).cumprod()
            self.long_short_net = self.long_short_net / self.long_short_net.iloc[0]
        else:
            self.long_short_ret = (
                self.group_rets[self.long_name]
                - self.group_rets[self.short_name]
                - 2 * self.trade_cost / self.freq
            )
            self.long_short_net = (self.long_short_ret + 1).cumprod()
            self.long_short_net = self.long_short_net / self.long_short_net.iloc[0]
        self.group_rets = pd.concat(
            [self.group_rets, self.long_short_ret.to_frame("long_short")], axis=1
        )
        self.group_nets = pd.concat(
            [self.group_nets, self.long_short_net.to_frame("long_short")], axis=1
        )
        self.long_short_comments = self.comments(
            self.long_short_net, self.long_short_ret
        )
        if print_comments:
            print(self.long_short_comments)
        if plt_plot:
            self.group_nets.plot(rot=60)
            plt.savefig(filename + ".png")
            plt.show()
        if plotly_plot:
            fig = pe.line(self.group_nets)
            filename_path = filename + ".html"
            pio.write_html(fig, filename_path, auto_open=True)


class pure_moonson(object):
    """行业轮动回测框架"""

    def __init__(self, fac, group_num=5):
        homeplace = HomePlace()
        pindu = (
            pd.read_feather(homeplace.daily_data_file + "各行业行情数据.feather")
            .set_index("date")
            .resample("M")
            .last()
        )
        rindu = pindu / pindu.shift(1) - 1
        self.rindu = rindu
        self.fac = fac
        self.group = self.get_groups(fac, group_num)
        print("未完工，待完善，暂时请勿使用⚠️")

    def get_groups(self, df, groups_num):
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
