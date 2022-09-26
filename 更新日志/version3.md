## 更新日志🗓 — v3

* v3.2.9 — 2022.09.26
> 1. 给pure_helper增加说明
> 1. 修复了以mysql分钟数据更新因子值的类pure_fall的出现重复数据的潜在bug
* v3.2.8 — 2022.09.26
> 1. 修复了用分钟数据计算因子值时，数据重复的潜在bug
> 1. 增加了将因子值改为分组组号的函数to_group
> 1. 增加了根据因子b对因子a进行排序，并在组内使用某种操作的类pure_helper
> 1. 对回测框架的pure_moonnight的输出因子值增加保护
* v3.2.7 — 2022.09.20
> 1. 修复了回测框架pure_moonnight不设置sheetname就无法回测的bug，改为不设置sheeetname就不会写入excel
> 1. 给因子后续必要测试follow_tests增加写入多头在主要指数上的超额净值序列
* v3.2.6 — 2022.09.19
> 1. 通过import pure_ocean_breeze导入库的时候，不再自动导入pure_ocean_breeze.state.states模块内的内容，可通过pure_ocean_breeze.states来调用
> 2. 新增了对因子一键进行必要后续测试的函数follow_tests，包括输出各个分组表现、与常用风格因子相关系数、barra纯净化、在沪深300、中证500、中证1000指数上的多空绩效和多头超额表现、在各个一级行业上的Rank IC值和各个一级行业买n只股票的超额表现
> 3. 在pure_ocean_breeze.state.states模块中，新增COMMENTS_WRITER和NET_VALUES_WRITER参数，用于管理全局所有的pu re_moonnight和follow_tests的绩效记录和净值序列的记录
> 4. 修复了更新日频数据函数database_update_daily_files中，读取旧数据部分的潜在bug
> 5. 删去了pure_ocean_breeze.labor.comment模块中的，输出十组每组绩效表现的函数comments_ten，给pure_moonnight类新增函数pure_moonnight.comments_ten()，用于输入十分组各组绩效表现
> 6. 修复了将因子限定在指数成分股内的函数daily_factor_on300500中，读取指数成分股数据时的潜在bug
> 7. 给讲因子限定在各个一级行业成分股内的函数daily_factor_on_industry，增加了申万一级行业和中信一级行业可选的参数
> 8. 将在各个一级行业上进行分组多空测试的函数group_test_on_swindustry更名为group_test_on_industry，并增加申万一级行业和中信一级行业可选的参数
> 9. 将在各个一级行业上进行Rank IC测试的函数rankic_test_on_swindustry更名为rankic_test_on_industry，并增加申万一级行业和中信一级行业可选的参数
> 10. 修复了在各个一级行业上进行购买n只股票的多头超额测试的函数long_test_on_industry内的中性化bug、读取中信哑变量的bug、股票上市天数的bug、计算收益率序列的bug、读取各个行业指数的bug、中信行业名称bug
> 11. 修复了行业市值中性化函数decap_industry中读取流通市值数据的bug
> 12. 修复了使用clickhouse或questdb的分钟数据更新因子值的类pure_fall_frequent更新因子值时，在每段第一个交易日时的bug
* v3.2.5 — 2022.09.16
> 1. 修复了读取日频数据函数read_daily由于索引名称更改导致的bug
> 1. 修复了缓存机制导致同一内核中，无法转换中信行业和申万行业的bug
> 1. 给用clickhouse的分钟数据计算因子值的类pure_fall_frequent增加了notebook进度条功能，当tqdm_inside指定为-1时，即使用tqdm.tqdm_notebook功能
* v3.2.4 — 2022.09.15
> 1. 改善了以clickhouse和questdb分钟数据计算因子的循环逻辑，将需要计算的时间拆分为多段相邻时间来计算，并补充了起始第一天的计算
> 1. 将保存最终因子值的函数database_save_final_factors增加了去除全空行的功能
* v3.2.3 — 2022.09.13
> 1. 修复了日频数据更新中的bug
> 2. 修复了中信一级行业成分股数据更新的bug
> 3. 将日频数据更新限制时间调整回了17点前

* v3.2.2 — 2022.09.13
>1. 将日频数据更新的限制时间点由17点前，改为18点前
>1. 修复了调用旧版本时legacy_version的bug
* v3.2.1 — 2022.09.13
>1. 在更新日频数据中加入17点前，仅更新至上一个交易日的限制
>1. 将更新申万一级行业成分股的函数更名为database_update_swindustry_member
>1. 修复了调用旧版本v3p1时的bug
* v3.2.0 — 2022.09.13
>1. 将以mat格式存储的文件，全部转化为feather格式
>1. read_daily函数不再对停牌日的价格进行处理
>1. 删去了更新日频数据的参数，改为只能更新到最新日期
>1. 删去了更新辅助字典的记录，改为从已有的数据中识别上次更新的截止日期
>1. 更新中加入了去重，避免重复更新引起的潜在bug
>1. 在储存最终因子值的函数database_save_final_factors中加入了去重
>1. 增加了中信一级行业成分股数据的函数
>1. 删去了初始化中关于辅助字典的内容，不再创建database_config.db
>1. 删去了初始化中关于日频数据截止日期和分钟数据截止日期的设置要求
>1. 新增在行业上的超额收益测试函数long_test_on_industry，可以选择申万或中信一级行业，默认使用中信
>1. 保留原有风格的行业超额收益测试函数long_test_on_swindustry和long_test_on_zxindustry
>1. 给行业市值中性化函数decap_industry增加中信和申万可选参数，默认使用中信
>1. 简化pure_moon/pure_moonnight回测框架的基础数据处理流程
>1. 给pure_moon/pure_moonnight回测框架的行业中性化部分，增加了中信和申万一级行业可选的参数，默认使用中信
>1. 给pure_moon回测框架增加了设置回测时用到的基础数据的函数set_basic_data
>1. 给pure_moonnight增加了输入基础数据的参数，包括上市天数、是否st、是否正常交易、复权开盘价、复权收盘价、月末流通市值
>1. 删去了pure_moon/pure_moonnight回测框架的by10参数，日频状态月频化时只能使用比较大小的方式
>1. 在旧版本合集legacy_version模块中，收录了最后一个可以处理mat文件的版本3.1.6，使用import pure_ocean_breeze.legacy_version.v3p1 as p即可调用3.1.6版本
* v3.1.6 — 2022.09.03
>1. 修复了用mysql更新因子值时的潜在bug
* v3.1.5 — 2022.09.03
>1. 修复了withs模块的bug
* v3.1.4 — 2022.09.03
>1. 将with模块改为withs模块，避免与关键字冲突
>1. 将更新的源限定为pypi
* v3.1.3 — 2022.09.03
>1. 加入了with模块，可以通过`from pure_ocean_breeze.with.requires import *`加载所有依赖库
>1. 将自动检查新版本改为了需要手动调用check_update()函数来检查
>1. 将行业市值中性化函数decap_industry()改为了可手动指定频率，如果未指定，再自动识别
>1. 修复了限定因子在申万一级行业上的函数daily_factor_on_swindustry()的bug
>1. 修复了使用mysql更新因子值时的表名bug
>1. 修复了使用clickhouse、questdb、postgresql数据库更新因子值时的潜在bug
* v3.1.2 — 2022.08.31
>1. 修复了导入时循环引用的bug
* v3.1.1 — 2022.08.30
>1. 增加了自动检测新版本的功能，在导入库时将自动检测并输出结果
>1. 增加了导入库的时间段统计，可使用函数show_use_times()查看当前设备各个时间段导入次数
>1. 增加了升级库的函数upgrade()，也可以简写为up()，将框架升级为最新版
>1. 将米筐rqdatac改为了非必要加载库
>1. 给read_daily函数增加了一键读入上市天数、流通市值、是否st股、交易状态是否正常的选项
>1. 给读入申万行业指数read_swindustry_prices和中信行业指数read_zxindustry_prices加入起始日期参数
>1. 给获取申万一级行业哑变量的函数get_industry_dummies增加起始日期参数
>1. 增加将一个因子值拆分在各个申万一级行业上的因子值的函数daily_factor_on_swindustry
>1. 增加分别在每个申万一级行业上测试因子分组回测表现的函数group_test_on_swindustry
>1. 增加专门计算因子值在各个申万一级行业上的Rank IC值，并绘制柱状图的函数rankic_test_on_swindustry
>1. 增加对每个申万一级行业成分股，使用某因子挑选出最多头的n值股票，考察其超额收益绩效、每月超额收益、每月每个行业的多头名单的函数long_test_on_swindustry
* v3.1.0 — 2022.08.28
>1. 增加了资金流相关数据，包括一键读入资金流数据的函数read_money_flow和更新数据的函数
>2. 调整了读取因子值和写入因子值函数所属的模块
* v3.0.9 — 2022.08.28
>1. 恢复了read_market函数，将一键读入wind全A指数改为一键读入中证全指000985.SH
* v3.0.8 — 2022.08.25
>1. 修复了更新宽基指数成分股的bug
* v3.0.7 — 2022.08.25
>1. 向数据库模块（pure_ocean_breeze.data.database）中增加了数据库元类，包含不同数据库通用的功能与属性
>1. 向数据库模块中增加使用postgresql与psycopg2引擎的数据库类，包含相关数据库通用的属性与连接方式
>1. 增加postgresql数据库与questdb数据库模块
>1. 统一化mysql数据库的命名
>1. 修复了识别代码为股票与基金代码的函数bug
>1. 增设同时更新clickhouse与questdb分钟数据的函数，以及单独更新questdb和单独更新postgresql的分钟数据的函数
>1. 删去了mysql每只股票/基金一张表的更新部分，仅保留每天一张表
>1. 将mysql的数据存储改为FLOAT，不再乘以100
>1. 以分钟数据计算因子值的部分，增设使用questdb和postgresql计算的选项
* v3.0.6 — 2022.08.21
>1. 修复了使用sql更新因子值的文件读入bug
* v3.0.5 — 2022.08.19
>1. 将旧版本模块与未来版本模块改为了须单独导入
>2. 向本地数据库增设了中信一级行业指数数据
>3. 增加了中信一级行业的米筐代码与行业名称之间的对应字典
* v3.0.4 — 2022.08.18
>1. 增设了旧版本模块（pure_ocean_breeze.legacy_version)，目前包括三个旧版本：
>>*  v2.6.9（发布时间2022-08-16）
>>*  v1.20.7（发布时间2022-07-11）
>>*  v1.10.7（发布时间2022-04-04)
>2. 增设了未来版本模块(pure_ocean_breeze.future_version)，目前包括两个部分：
>>* pure_ocean_breeze.future_version.half_way（尚未完工的部分）
>>* pure_ocean_breeze.future_version.in_thoughts（仍主要在构思或推敲阶段的部分）
* v3.0.3 — 2022.08.18
>1. 调整了部分模块的名称，使其可以被mkdocstrings识别
>1. 增设了主流宽基指数wind代码与简称的对应字典
>1. 增设专门v2模块，以便在v3版本中调用旧版本代码
* v3.0.2 — 2022.08.18
>1. 剔除常用风格因子的模块（pure_coldwinter/pure_snowtrain)可以增加自定义因子或选择不剔除某些因子
>1. 发送邮件的模块，改为可以不带附件，发送纯文本邮件
* v3.0.1 — 2022.08.17
>1. 删去了回测框架中读入最高价、最低价、成交量和换手率的部分
>1. 将申万行业哑变量的读入时间提前，从而给回测提速60%
* v3.0.0 — 2022.08.17
>1. 上线了[说明文档](https://chen-001.github.io/pure_ocean_breeze/)
>2. 将v2中的pure_ocean_breeze模块拆分为不同功能的几个模块
>>* initialize （初始化）
>>* state （配置&参数）
>>* data （数据）
>>* labor （加工&测试&评价）
>>* mail（通讯）
>3. 修复了主要指数成分股处理的bug，并将其改为日频
>4. 增加了国证2000成分股的哑变量
>5. 删去了初始化中的分钟数据文件路径