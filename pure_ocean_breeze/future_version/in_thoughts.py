__updated__ = "2022-11-04 16:42:45"

import copy
import numpy as np
import pandas as pd
from pure_ocean_breeze.data.read_data import read_daily
from pure_ocean_breeze.data.tools import multidfs_to_one


class pure_wood(object):
    """一种因子合成的方法，灵感来源于adaboost算法
    adaboost算法的精神是，找到几个分类效果较差的弱学习器，通过改变不同分类期训练时的样本的权重，
    计算每个学习器的错误概率，通过线性加权组合的方式，让弱分类期进行投票，决定最终分类结果，
    这里取通过计算各个因子多头或空头的错误概率，进而通过错误概率对因子进行加权组合，
    此处方式将分为两种，一种是主副因子的方式，另一种是全等价的方式，
    主副因子即先指定一个主因子（通常为效果更好的那个因子），然后指定若干个副因子，先计算主因子的错误概率，
    找到主因子多头里分类错误的部分，然后通过提高期加权，依次计算后续副因子的错误概率（通常按照因子效果从好到坏排序），
    最终对依次得到的错误概率做运算，然后加权
    全等价方式即不区分主副因子，分别独立计算每个因子的错误概率，然后进行加权"""

    def __init__(self, domain_fac: pd.DataFrame, subdomain_facs: list, group_num=10):
        """声明主副因子和分组数"""
        self.domain_fac = domain_fac
        self.subdomain_facs = subdomain_facs
        self.group_num = group_num
        opens = read_daily(open=1).resample("M").first()
        closes = read_daily(close=1).resample("M").last()
        self.ret_next = closes / opens - 1
        self.ret_next = self.ret_next.shift(-1)
        self.domain_fac = (
            self.domain_fac.T.apply(
                lambda x: pd.qcut(x, group_num, labels=False, duplicates="drop")
            ).T
            + 1
        )
        self.ret_next = (
            self.ret_next.T.apply(
                lambda x: pd.qcut(x, group_num, labels=False, duplicates="drop")
            ).T
            + 1
        )
        self.subdomain_facs = [
            i.T.apply(
                lambda x: pd.qcut(x, group_num, labels=False, duplicates="drop")
            ).T
            + 1
            for i in self.subdomain_facs
        ]
        self.get_all_a()
        self.get_three_new_facs()

    def __call__(self, *args, **kwargs):
        return copy.copy(self.new_facs)

    def get_a_and_new_weight(self, n, fac, weight=None):
        """计算主因子的权重和权重矩阵"""
        fac_at_n = (fac == n) + 0
        ret_at_n = (self.ret_next == n) + 0
        not_nan = fac_at_n + ret_at_n
        not_nan = not_nan[not_nan.index.isin(fac_at_n.index)]
        not_nan = (not_nan > 0) + 0
        wrong = ((ret_at_n - fac_at_n) > 0) + 0
        wrong = wrong[wrong.index.isin(fac_at_n.index)]
        right = ((ret_at_n - fac_at_n) == 0) + 0
        right = right[right.index.isin(fac_at_n.index)]
        wrong = wrong * not_nan
        right = right * not_nan
        wrong = wrong.dropna(how="all")
        right = right.dropna(how="all")
        if isinstance(weight, pd.DataFrame):
            e_rate = (wrong * weight).sum(axis=1)
            a_rate = 0.5 * np.log((1 - e_rate) / e_rate)
            wrong_here = -wrong
            g_df = multidfs_to_one(right, wrong_here)
            on_exp = (g_df.T * a_rate.to_numpy()).T
            with_exp = np.exp(on_exp)
            new_weight = weight * with_exp
            new_weight = (new_weight.T / new_weight.sum(axis=1).to_numpy()).T
        else:
            e_rate = (right.sum(axis=1)) / (right.sum(axis=1) + wrong.sum(axis=1))
            a_rate = 0.5 * np.log((1 - e_rate) / e_rate)
            wrong_here = -wrong
            g_df = multidfs_to_one(right, wrong_here)
            on_exp = (g_df.T * a_rate.to_numpy()).T
            with_exp = np.exp(on_exp)
            new_weight = with_exp.copy()
            new_weight = (new_weight.T / new_weight.sum(axis=1).to_numpy()).T
        return a_rate, new_weight

    def get_all_a(self):
        """计算每个因子的a值"""
        # 第一组部分
        one_a_domain, one_weight = self.get_a_and_new_weight(1, self.domain_fac)
        one_a_list = [one_a_domain]
        for fac in self.subdomain_facs:
            one_new_a, one_weight = self.get_a_and_new_weight(1, fac, one_weight)
            one_a_list.append(one_new_a)
        self.a_list_one = one_a_list
        # 最后一组部分
        end_a_domain, end_weight = self.get_a_and_new_weight(
            self.group_num, self.domain_fac
        )
        end_a_list = [end_a_domain]
        for fac in self.subdomain_facs:
            end_new_a, end_weight = self.get_a_and_new_weight(
                self.group_num, fac, end_weight
            )
            end_a_list.append(end_new_a)
        self.a_list_end = end_a_list

    def get_three_new_facs(self):
        """分别使用第一组加强、最后一组加强、两组平均的方式结合"""
        one_fac = sum(
            [
                (i.iloc[1:, :].T * j.iloc[:-1].to_numpy()).T
                for i, j in zip(
                    [self.domain_fac] + self.subdomain_facs, self.a_list_one
                )
            ]
        )
        end_fac = sum(
            [
                (i.iloc[1:, :].T * j.iloc[:-1].to_numpy()).T
                for i, j in zip(
                    [self.domain_fac] + self.subdomain_facs, self.a_list_end
                )
            ]
        )
        both_fac = one_fac + end_fac
        self.new_facs = [one_fac, end_fac, both_fac]


class pure_fire(object):
    """一种因子合成的方法，灵感来源于adaboost算法
    adaboost算法的精神是，找到几个分类效果较差的弱学习器，通过改变不同分类期训练时的样本的权重，
    计算每个学习器的错误概率，通过线性加权组合的方式，让弱分类期进行投票，决定最终分类结果，
    这里取通过计算各个因子多头或空头的错误概率，进而通过错误概率对因子进行加权组合，
    此处方式将分为两种，一种是主副因子的方式，另一种是全等价的方式，
    主副因子即先指定一个主因子（通常为效果更好的那个因子），然后指定若干个副因子，先计算主因子的错误概率，
    找到主因子多头里分类错误的部分，然后通过提高期加权，依次计算后续副因子的错误概率（通常按照因子效果从好到坏排序），
    最终对依次得到的错误概率做运算，然后加权
    全等价方式即不区分主副因子，分别独立计算每个因子的错误概率，然后进行加权"""

    def __init__(self, facs: list, group_num=10):
        """声明主副因子和分组数"""
        self.facs = facs
        self.group_num = group_num
        opens = read_daily(open=1).resample("M").first()
        closes = read_daily(close=1).resample("M").last()
        self.ret_next = closes / opens - 1
        self.ret_next = self.ret_next.shift(-1)
        self.ret_next = (
            self.ret_next.T.apply(
                lambda x: pd.qcut(x, group_num, labels=False, duplicates="drop")
            ).T
            + 1
        )
        self.facs = [
            i.T.apply(
                lambda x: pd.qcut(x, group_num, labels=False, duplicates="drop")
            ).T
            + 1
            for i in self.facs
        ]
        self.get_all_a()
        self.get_three_new_facs()

    def __call__(self, *args, **kwargs):
        return copy.copy(self.new_facs)

    def get_a(self, n, fac):
        """计算主因子的权重和权重矩阵"""
        fac_at_n = (fac == n) + 0
        ret_at_n = (self.ret_next == n) + 0
        not_nan = fac_at_n + ret_at_n
        not_nan = not_nan[not_nan.index.isin(fac_at_n.index)]
        not_nan = (not_nan > 0) + 0
        wrong = ((ret_at_n - fac_at_n) > 0) + 0
        wrong = wrong[wrong.index.isin(fac_at_n.index)]
        right = ((ret_at_n - fac_at_n) == 0) + 0
        right = right[right.index.isin(fac_at_n.index)]
        wrong = wrong * not_nan
        right = right * not_nan
        wrong = wrong.dropna(how="all")
        right = right.dropna(how="all")
        e_rate = (right.sum(axis=1)) / (right.sum(axis=1) + wrong.sum(axis=1))
        a_rate = 0.5 * np.log((1 - e_rate) / e_rate)
        return a_rate

    def get_all_a(self):
        """计算每个因子的a值"""
        # 第一组部分
        self.a_list_one = [self.get_a(1, i) for i in self.facs]
        # 最后一组部分
        self.a_list_end = [self.get_a(self.group_num, i) for i in self.facs]

    def get_three_new_facs(self):
        """分别使用第一组加强、最后一组加强、两组平均的方式结合"""
        one_fac = sum(
            [
                (i.iloc[1:, :].T * j.iloc[:-1].to_numpy()).T
                for i, j in zip(self.facs, self.a_list_one)
            ]
        )
        end_fac = sum(
            [
                (i.iloc[1:, :].T * j.iloc[:-1].to_numpy()).T
                for i, j in zip(self.facs, self.a_list_end)
            ]
        )
        both_fac = one_fac + end_fac
        self.new_facs = [one_fac, end_fac, both_fac]
