__updated__ = "2023-10-14 15:32:40"

# 申万一级行业的代码与名字对照表
INDUS_DICT = {
    k: v
    for k, v in zip(
        [
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
        [
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
    )
}

# 常用宽基指数的代码和名字
INDEX_DICT = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "399303.SZ": "国证2000",
}

# 中信一级行业代码和名字对照表
ZXINDUS_DICT = {
    "CI005001.INDX": "石油石化",
    "CI005002.INDX": "煤炭",
    "CI005003.INDX": "有色金属",
    "CI005004.INDX": "电力及公用事业",
    "CI005005.INDX": "电力及公用事业",
    "CI005005.INDX": "钢铁",
    "CI005006.INDX": "基础化工",
    "CI005007.INDX": "建筑",
    "CI005008.INDX": "建材",
    "CI005009.INDX": "轻工制造",
    "CI005010.INDX": "机械",
    "CI005011.INDX": "电力设备及新能源",
    "CI005012.INDX": "国防军工",
    "CI005013.INDX": "汽车",
    "CI005014.INDX": "商贸零售",
    "CI005015.INDX": "消费者服务",
    "CI005016.INDX": "家电",
    "CI005017.INDX": "纺织服装",
    "CI005018.INDX": "医药",
    "CI005019.INDX": "食品饮料",
    "CI005020.INDX": "农林牧渔",
    "CI005021.INDX": "银行",
    "CI005022.INDX": "非银行金融",
    "CI005023.INDX": "房地产",
    "CI005024.INDX": "交通运输",
    "CI005025.INDX": "电子",
    "CI005026.INDX": "通信",
    "CI005027.INDX": "计算机",
    "CI005028.INDX": "传媒",
    "CI005029.INDX": "综合",
    "CI005030.INDX": "综合金融",
}
