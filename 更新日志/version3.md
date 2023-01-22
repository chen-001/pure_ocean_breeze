## 更新日志🗓 — v3

* v3.7.1 — 2023.1.22
> 1. 给MetaSQLDriver的get_data增加了尝试10次后再报错的设定，以避免偶然出现的连接失败错误
> 1. 修复了pure_fall_frequent中使用questdb调取数据的命令语句的bug
> 1. 修复了pure_fall_frequent中使用questdb调取数据的数据类型的bug
* v3.7.0 — 2023.1.20
> 1. 修复了Questdb通过csv写入表格时，进程占用文件，导致无法删除的bug
> 1. 给read_market、read_index_single、database_update_minute_data_to_clickhouse_and_questdb、database_update_minute_data_to_questdb增加了自动调整web_port参数值的机制
> 1. 给pure_fall_frequent增加了questdb_web_port参数值，用于指定本台设备questdb的web_port值
> 1. 修复了pure_rollingols中betas属性显示异常的bug
* v3.6.9 — 2023.1.19
> 1. 对drop_duplicates_index函数，增加了保留原来index名字的功能
> 1. 删去了pure_moon回测中的弹窗提示
> 1. 补充了一些依赖库
> 1. 修正了一些函数说明
* v3.6.8 — 2023.1.13
> 1. 修复了初始化存在的bug
* v3.6.7 — 2023.1.13
> 1. 修复了Questdb中关于web_port参数的bug
> 1. 修复了database_update_minute_data_to_questdb的bug
>
* v3.6.6 — 2023.1.10
> 1. 新增了database_update_industry_rets_for_stock函数，用于生成每只股票当天对应的一级行业的收益率
> 1. 给read_daily函数新增了swindustry_ret和zxindustry_ret参数，可以读取每只股票当天对应的一级行业的收益率数据
* v3.6.5 — 2023.1.6
> 1. 给Questdb初始化新增了web_port参数，用于表示控制台的端口号
> 1. 给read_daily函数新增了money参数用于读取每日个股成交额、illiquidity参数用于读取每日个股非流动性
> 1. 新增了database_update_illiquidity函数用于更新每天非流动性数据
> 1. 删去了pure_moon中的select_data_time方法，在set_basic_data函数中新增了基础数据为None时从本地读入的方法
> 1. 优化了pure_moonnight因子值时间超过基础数据时间时，结果的展示方式
> 1. 优化了pure_moonnight的运算逻辑，对回测进行提速，并恢复了time_start和time_end参数的使用，可以为每次回测单独设定回测区间
> 1. 修复了do_on_dfs装饰器在仅作用于一个目标时，参数不生效的bug
* v3.6.4 — 2022.12.29
> 1. 新增了do_on_dfs装饰器，用于将一个作用于单个dataframe的函数，改造为可以分别对多个dataframe运算，dataframe须处于第一个参数的位置，此外如果对每个dataframe，后续的某个参数各不相同，可使用列表依次输入。
> 2. 修复了clip函数的bug
> 3. 新增了judge_factor_by_third函数，用于依据第三个指标的正负，对两个因子进行筛选合成，第三个指标为正则取因子1的值，为负则取因子2的值
> 4. 给pure_fall_frequent的run函数新增了many_days参数，用于对每天取之前的n天的分钟数据来计算因子，当groupby_target指定为`['date','code']`或`['code']`时，则不涉及截面，每只股票各自算自己的因子值，返回值应为float、list、tuple；当groupby_target指定为`[]`时，则取多天的截面数据，返回值应为pd.Series（index为股票代码，values为因子值），或多个这样的pd.Series
> 5. 给pure_fall_frequent新增了drop_table函数，用于删去计算到一半被打断后，不想要的questdb暂存表格
> 6. 新增get_group函数，使用groupby的方法，将一组因子值改为截面上的分组值，此方法相比qcut的方法更加稳健，但速度更慢一些
* v3.6.3 — 2022.12.25
> 1. 修复了database_read_final_factors函数读取周频因子数据的bug
> 1. 修复了merge_many及其他相关函数的bug
> 1. 将corr_two_daily、com_two_daily、func_two_daily的默认并行数改为了1
> 1. 将show_corrs函数不再使用print打印相关系数表格，而是直接返回，此外print_bool参数的函数改为绝对是否将返回值的数值变为百分数
> 1. 修复了show_corrs_with_old时的数字混乱bug
* v3.6.2 — 2022.12.18
> 1. 给read_daily函数增加了vol_daily参数，用于读取使用分钟收益率的标准差计算的每日波动率
> 1. 使read_daily读出的数据不再包含全为空值的行
> 1. 修复了因子标号跳跃导致的show_corrs_with_old函数显示不全的bug
* v3.6.1 — 2022.12.14
> 1. 修复了drop_duplicates_index的bug
> 1. 给pure_moonnight增加了without_breakpoint参数，用于控制iplot画图忽略空格
> 1. 新增了和pure_moon功能一模一样的pure_week，并修复了在同一进程中，不能使用pure_moonnight同时进行月频回测和周频回测的bug
* v3.6.0 — 2022.12.13
> 1. 读取市场指数数据的函数read_market不再限定只能读取中证全指的数据，改为可以通过market_code参数读取任何指数数据
> 1. 修复了read_market函数中使用questdb读取最高价、最低价、开盘价数据的bug，以及不能指定起始时间的bug
> 1. 给read_index_single函数增加了说明
> 1. 向读取h5相关文件函数添加了下线警告
> 1. 给convert_code函数增加了把wind代码转化为米筐代码的功能
> 1. 修复了drop_duplicates_index函数在某些情况下可能出现bug的问题
> 1. 新增了对因子去极值的函数clip，以及三个细分函数clip_mad、clip_three_sigma、clip_percentile
> 1. 修复了pure_moon绩效输出到excel中时，信息比率写入错误的bug
> 1. 给pure_fall_frequent新增了project参数，用于标记每个因子所属于的项目，便于管理
> 1. 修复了pure_fall_frequent打断后重新运行后可能在的bug
> 1. 删去了pure_fall_frequent中关于“共x段”的显示
* v3.5.9 — 2022.11.29
> 1. 修复了读取不复权价格数据时的bug
>
> 2. 新增了将所有因子截面上减去最小值，使其变为非负数的函数all_pos
>
> 3. 新增了对日票或月频因子值剔除st股、停牌股、上市不足60天的函数remove_unavailable
>
> 4. pure_moon类中set_basic_data函数的参数值不再必填
>
> 5. 修复了pure_moon不做行业市值中性化时，factors_out属性的columns异常的问题
>
> 6. 将pure_moon回测的绩效指标的`RankIC均值t值`改为了`RankIC.t`
>
> 7. pure_moon回测新增了6个绩效指标
> > * self.factor_cover：因子覆盖率，为因子的数量与复权开盘价数据的数量的比值
> > * self.pos_neg_rate：因子正值占比，为原始因子值中，为正数的数量占非零值数量的比值
> > * self.factor_cross_skew：因子截面偏度，为原始因子值每个截面的偏度序列的均值，为0则为正态分布，为正则右偏（右边尾巴长），小于零则左偏（左边尾巴长）
> > * self.factor_cross_skew_after_neu：中性化后偏度，对因子进行行业市值中性化之后，平均截面偏度
> > * self.corr_itself：一阶自相关性，原始因子值与其上一期的因子值之间的对应期截面spearman相关系数的均值
> > * self.corr_itself_shift2：二阶自相关性，原始因子值与其上两期的因子值之间的对应期截面spearman相关系数的均值
> 8. 取消了原来的每月换手率变化曲线，新增了每月原始因子值的截面标准差的柱状图
> 8. pure_moon中的回测结果使用cufflinks画图时（即iplot为1时），默认不再显示图例，即ilegend默认值改为了0
> 8. 调整了pure_moon类中set_basic_data函数的形参名称
> 8. 修复了pure_fall_frequent中，暂存至questdb时，表名中带有特殊符号会无法删除表格的bug
>
* v3.5.8 — 2022.11.19
> 1. 给database_read_final_factors和database_save_final_factors新增了`freq`参数，可以指定为'月'或'周'，即可存入月频因子数据（滚动20天）和周频因子（滚动5天）数据
> 2. 新增了`计算连续期数`函数，可以用于计算某个指标连续大于或连续小于某个阈值的期数
> 3. 优化了test_on_index_four中，对多头超额净值曲线的显示
* v3.5.7 — 2022.11.17
> 1. 新增了将日频因子进行面板操作以月度化的函数get_fac_via_corr、get_fac_cross_via_func
> 1. 给pure_moon的评价指标新增了RankIC胜率
> 1. 修复了在分钟数据某日出现特殊情况时，会导致pure_fall_frequent在写入questdb时发生错误的bug
* v3.5.6 — 2022.11.14
> 1. 删去了`Homeplace.__slots__`中的minute_data_file、daily_enddate、minute_enddate，以修复新用户初始化后导入失败的bug
* v3.5.5 — 2022.11.13
> 1. 删去了change_index_name函数，可以通过形如`df.index.name='date'`的代码直接指定
> 1. 在write_data模块中，新增了连接米筐失败时，输出失败原因，提供是否等待的选项，并设置了在30秒、60秒、60秒内重连三次的尝试
> 1. 修复了rankic_test_on_industr、follow_tests函数在不指定comments_writer和net_values_writer时报错的bug
> 1. 在pure_ocean_breeze.withs.requires中连接米筐失败时，将给出原因
* v3.5.4 — 2022.11.11
> 1. 给merge_many增加了how参数，可以指定拼接的方式，默认为outer
> 1. 新增zip_many_dfs函数，用于将多个dataframe拼在一起，相同index和columns指向的那个values，连在一起，组成列表
> 1. 新增get_values函数，用于一次性取出一个values为列表的dataframe的所有值，分别设置一个新的dataframe
> 1. 给comment_on_rets_and_nets和comment_on_twins新增参数counts_one_year，用于设置序列的频率
> 1. 新增了boom_fours函数，用于对一个列表的因子值分别算boom_four
* v3.5.3 — 2022.11.6
> 1. 将所有通过is_notebook判断使用何种进度条的函数都改为了通过tqdm.auto模块自动判断
> 1. 将重复更新日频数据database_update_daily_files函数设置为不再报错
> 1. 修复了pure_fall_frequent只更新一天数据时不写入questdb的bug
> 1. 修复了使用pure_fall_frequent进行截面操作时，使用for_cross_via_zip装饰器，只返回一个Series时的bug
> 1. 将pure_ocean_breeze.withs.requires中导入的进度条模块改为tqdm.auto
> 1. 向依赖库中新增了tradetime，删去了SciencePlots、alphalens和cachier
* v3.5.2 — 2022.11.6
> 1. pure_moon和pure_moonnight新增freq参数，可以选择'M'，或'W'，进行月频测试或周频测试
> 1. 新增frequency_controller类，对回测中不同频率的操作和参数进行控制
> 1. 修复了预存储月度交易状态和st状态的错误
> 1. 优化了pure_moon中部分算法
> 1. pure_ocean_breeze.withs.requires中新增了`import tradetime as tt`用于进行交易日历上的相关操作
* v3.5.1 — 2022.11.5
> 1. 修复了database_update_minute_data_to_clickhouse_and_questdb中的get_price接口变化导致的bug
> 1. 删去了plt.style的设定代码
> 1. 修复了pure_ocean_breeze.withs.requires中导入Iterable的bug
* v3.5.0 — 2022.11.5
> 1. 替换了所有的feather文件读写和存储，改为parquet格式，优化了索引相关的操作
> 1. 优化了pure_moon中关于因子处理的步骤
> 1. legacy_version中收录了v3p4，即3.4.8版本，为最后一个通过feather文件读写的版本
> 1. 新增了feather_to_parquet_all函数，一键将数据库中的所有feather文件都转化为parquet文件
> 1. 删去了依赖包feather，新增了依赖包pyarrow和clickhouse_sqlalchemy
* v3.4.8 — 2022.11.4
> 1. 修复了pure_moon市值加权回测的bug
> 1. 优化了pure_moon的on_paper参数下，学术化评价指标的内容
> 1. 给pure_moonnight增加了
> > * 识别pure_ocean_breeze.state.states中的ON_PAPER参数，用来对环境中所有的回测展示学术化的评价指标
> > * 识别pure_ocean_breeze.state.states中的MOON_START参数，用来对环境中所有的回测的因子规定起点
> > * 识别pure_ocean_breeze.state.states中的MOON_END参数，用来对环境中所有的回测的因子规定终点
> 4. 给pure_ocean_breeze.state.states中增加了ON_PAPER、MOON_START、MOON_END全局参数
> 4. 新增了feather_to_parquet文件，可以将某一文件夹下的所有feather文件转化为parquet文件
> 4. 修复了pure_cloud中的bug
> 4. 修复了pure_ocean_breeze.future_version.in_thoughts模块的导入错误
> 4. 扩展了pure_moonnight的输入因子的类型范围，当前pure_fallmount、pure_snowtrain和dataframe都可以直接输入
* v3.4.7 — 2022.11.4
> 1. 删去了h5py这一依赖项，改为只有在调用read_h5和read_h5_new函数时才会import，并从pure_ocean_breeze.withs.requires中剔除
> 1. 修复了collections中Iterable加载在python3.11版本中的bug
> 1. 修复了pure_fall_frequent.for_cross_via_zip装饰器的bug
> 1. 剔除了对pretty_errors的依赖项
* v3.4.6 — 2022.11.2
> 1. 给read_money_flow增加了whole参数，用于读入当天各类型投资者买卖总量
> 1. 新增了same_columns和same_index函数，取很多dataframe，都只保留共同的columns或index的部分
> 1. 新增了show_x_with_func函数，用于考察两个因子同期截面上的某种关系，并返回一个时间序列
> 1. 新增了show_cov和show_covs函数，考察两个因子或多个因子截面协方差关系
> 1. 将pure_moon修改为，默认不再读入申万一行业哑变量数据，只读入中信一级行业哑变量数据
> 1. 优化了pure_moon回测结果展示图的位置
> 1. 修复了pure_fall_frequent进行截面计算时的bug
> 1. 给pure_fall_frequent新增了for_cross_via_str和for_cross_via_zip装饰器，用于简化截面计算时的对返回值处理的代码
* v3.4.5 — 2022.11.1
> 1. 修复了read_daily中读取不复权最低价时的bug
> 1. 修复了database_update_daily_files更新st股哑变量时的bug
> 1. 修复了pure_moon读取月度状态文件的bug
> 1. 修复了pure_fall_frequent计算单一一天因子值时的bug
> 1. 给pure_ocean_breeze.withs.requires模块中增加了time库
* v3.4.4 — 2022.10.31
> 1. 修复了test_on_index_four使用matplotlib画图不显示的bug
> 1. 新增了择时回测框架pure_star
* v3.4.3 — 2022.10.31
> 1. 给read_index_three增加了国证2000指数的行情数据
> 1. 给make_relative_comments增加了gz2000参数，用于计算相对国证2000指数的超额收益；增加了show_nets参数，用于同时返回多头超额评价指标和超额净值数据
> 1. 给make_relative_comments_plot增加了gz2000参数，用于绘制相当于国证2000指数的超额净值走势图
> 1. 给show_corr增加了show_series参数，将返回值从相关系数的均值改为了相关系数的序列，并取消绘图
> 1. 给pure_moon和pure_moonnight增加了swindustry_dummy和zxindustry_dummy参数，用于自己输入申万一级行业哑变量数据和中信一级行业哑变量数据
> 1. 修复了不同回测结果的IC序列相同的bug
> 1. 给pure_fall_frequent增加了ignore_history_in_questdb的参数，用于被打断后，忽略在questdb中的暂存记录，重新计算；新增了groupby_target参数，用于指定groupby分组计算因子值时，分组的依据（即df.groupby().apply()中groupby里的参数），此改进使得可以进行截面上的构造和计算
> 1. 给test_on_300500增加了gz2000参数，用于测试在国证2000成分股内的多空和多头超额效果
> 1. 恢复了test_on_index_four中的gz2000参数，用于测试国证2000成分股内的效果；优化了多头超额的绩效展示，整合了超额净值曲线图与一张上，使用cufflinks进行显示（可以通过iplot参数关闭cufflinks显示）
* v3.4.2 — 2022.10.29
> 1. 新增了is_notebook函数，可以判断当前环境是否为notebook
> 1. 将除write_data模块以外，其他模块的进度条，都改为可以自动识别环境为notebook，如果是notebook，则自动使用tqdm_notebook的进度条
* v3.4.1 — 2022.10.26
> 1. 给func_two_daily、corr_two_daily、cov_two_daily增加了history参数，用于将计算出的结果记录在本地
> 1. 给show_corrs、show_corrs_with_old函数增加了method参数，可以修改求相关系数的方式
> 1. 暂时删去了test_on_300500的国证2000的参数
> 1. 给test_on_300500和test_on_index_four新增了iplot参数，决定是否使用cufflinks画图
* v3.4.0 — 2022.10.25
> 1. 修复了拼接多个dataframe的函数merge_many中的bug
> 1. 修复了导入process模块时的bug
> 1. 新增了同时测试因子在单个宽基指数成分股上的多空和多头超额表现的函数test_on_3005000
> 1. 新增了同时测试因子在4个宽基指数成分股上的多空和多头超额表现的函数test_on_index_four
* v3.3.9 — 2022.10.24
> 1. 修复了使用clickhouse中get_data时的连接bug
> 1. 修复了拼接多个dataframe的函数merge_many中的bug
> 1. 给func_two_daily、corr_two_daily增加了n_jobs参数，用于决定并行数量
> 1. 新增了滚动求两因子协方差的函数cov_two_daily
> 1. 新增了求目标因子与已发研报因子之间的相关系数的函数show_corrs_with_old
> 1. 修复了pure_fall_frequent计算因子值被打断后，从questdb读取已经计算的因子值时潜在的bug
* v3.3.8 — 2022.10.21
> 1. 对clickhouse、questdb、postgresql数据库的get_data方法增加了只获取np.ndarray的参数
> 1. 给pure_moon增加wind_out属性，用于输出每个时期股票所属分组
> 1. 将pure_fall_frequent中，计算单日因子值的进度条删去
* v3.3.7 — 2022.10.10
> 1. 修复了使用pure_fall（mysql）的分钟数据，更新因子值时的，读取之前因子数据的bug
* v3.3.6 — 2022.10.09
> 1. 新增了剔除北交所因子数据的函数debj
> 1. 新增了对因子做横截面zscore标准化的函数standardlize
> 1. 新增了统计dataframe中有多少（非0）非空数据的函数count_value
> 1. 优化了检测dataframe中是否存在空值的函数detect_nan的计算方法
> 1. 新增了使用若干因子对目标因子正交化的函数de_cross
> 1. 对pure_moon和pure_moonnight新增了使用cufflinks展示回测结果和绘图的参数iplot和ilegend，优化了结果展示的样式
> 1. 修复了pure_fama中，不包含市场因子时的潜在bug
> 1. 对pure_fama中的coefficients属性进行保护
> 1. 新增了pure_rollingols类，用于对若干个因子，对应股票下，进行固定时间窗口的滚动回归
> 1. 调整了一些工具函数的分类，以减少循环引用bug的可能性
* v3.3.5 — 2022.10.07
> 1. 给pure_fall_frequent增加了中途写入questdb防止运算被打断的功能，并可以在打断后通过识别questdb中的数据继续计算，该临时表将在运算全部完成，并成功写入feather文件后删除
> 1. 在Questdb的写入函数write_via_csv的临时csv文件名中，加入了随机数，以避免同时写入文件时的冲突
> 1. 给drop_duplicates_index增加说明，并移至data.tools模块
> 1. 修复了一个讲分钟数据写入postgresql的bug
> 1. 修复了市值中性化函数decap，读取流通市值数据的bug
> 1. 给pure_moon模块的初始化函数参数no_read_indu增加默认参数0，以修复条件双变量排序类pure_newyear的bug
> 1. 修复了pure_fall_frequent中，tqdm_inside不为1时，无法计算第一个交易日的数据的bug
* v3.3.4 — 2022.10.06
> 1. 删去了回测类pure_moon和pure_moonnight中起始日期startdate参数，以避免输入因子值起点不同，会导致缓存失效的bug
> 1. 优化了pure_fama的参数逻辑，在输入add_markert_series参数时，自动将add_market参数指定为1
> 1. 新增了merge_many函数，将多个index为时间，columns位股票代码的dataframe拼接在一起，变成一个长表
> 1. 新增函数func_two_daily，用于对两个index为时间，columns为股票代码的dataframe，每只股票下，各自沿着时间序列，做某个函数操作，最终得到一个index为时间，columns为股票代码的dataframe
> 1. 新增func_two_daily的特例函数，corr_two_daily，求两个因子同一股票滚动窗口下的时序相关系数
* v3.3.3 — 2022.10.01
> 1. 将读取300、500、1000指数的行情read_index_three改为从分钟数据读取
> 1. 给读取市场行情（中证全指）行情read_market增加从questdb读取
> 1. 删除了pure_fall_frequent中使用postgresql中分钟数据计算因子值的选项
> 1. 修复了pure_fall_frequent中使用questdb中的分钟数据计算因子值的部分
* v3.3.2 — 2022.10.01
> 1. 修复了读取隔夜收益率的bug
> 1. 将更新特质收益率数据的起始时间点改为2010年1月1日，并修复了其中的bug
> 1. 给pure_moon和pure_moonnight增加了no_read_indu参数，使回测时不必读入行业哑变量数据，便于调试
> 1. 给pure_moon和pure_moonnight增加了only_cap参数，使回测时只做市值中性化，而不做行业中性化
> 1. 优化了pure_moon和pure_moonnight的参数逻辑，当neutralize和boxcox均为0时，自动开启no_read_indu参数；当no_read_indu和only_cap任一为0时，自动开启另一个参数
* v3.3.1 — 2022.10.01
> 1. 给一键导入库的requires中，增加了import pyfinance.ols as go
> 1. 增加了用于fama三因子与特质收益率相关的类pure_fama，可以计算各期因子收益率、个股各期特质收益率、个股各期因子暴露、超额收益率等内容
> 1. 更新每日数据函数database_update_daily_files增加了更新每日的市盈率pe和市净率pb的数据
> 1. 新增更新每日以20日回归，市场收益率、流通市值分三份、市净率分三份，计算的特质收益率数据
> 1. 一键读取日频数据函数read_daily新增参数ret（日间收益率）、ret_inday（日内收益率）、ret_night（隔夜收益率）、vol（滚动20日波动率）、vol_inday（滚动20日日内收益率波动率）、vol_night（滚动20日隔夜收益率波动率）、swing（振幅）、pb（市净率）、pe（市盈率）、iret（20日回归，市场、流通市值、市净率三因子特质收益率）、ivol（滚动20日，20日回归，市场、流通市值、市净率三因子特质波动率）
> 1. 增加了将dataframe第一列设置为index的函数set_index_first
> 1. 增加了修改index的名字的函数change_index_name
> 1. 丰富了回测类pure_moon和pure_moonnight的评价指标及绘图，新增属性self.ics（ic时间序列）、self.rankics（rankic时间序列）、factor_turnover_rates（每月换手率时间序列）、factor_turnover_rate（平均每月换手率）、group_rets_std（每组组内收益率的标准差）、group_rets_stds（每组组内收益率的标准差的时间序列），新增Rank IC时序图和换手率时序图，美化了评价指标的输出形式
* v3.3.0 — 2022.09.26
> 1. 修复了单独更新questdb内分钟数据的函数database_update_minute_data_to_questdb中的bug
> 1. 新增了依据index去重的函数drop_duplicates_index
> 1. 修复了更新日频数据可能重复的潜在bug
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