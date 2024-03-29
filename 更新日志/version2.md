## 更新日志🗓 — v2

* v2.6.7 — 2022.08.13
>1. 修复了基于clickhouse计算分钟数据因子，天数是10的倍数时的bug
>2. 调整画图风格为science+no_latex，及其他设置
>3. 增设读取初级因子功能
>4. 修改因子值排序
>5. 读取最终因子值函数，固定为返回元组
* v2.6.6 — 2022.08.06
>1. 使用black风格格式化了代码，增加可读性
>2. github自动更新至pypi
* v2.6.5 — 2022.08.05
>1. 修复了读取各行业行情数据的bug
>2. 暂时回退了minute_data_file路径参数，修复了pure_fall的bug
* v2.6.4 — 2022.08.02
>1. 修复了以mysql更新分钟因子的bug
>2. 改善了指数超额收益起点时间的运算逻辑
>3. 修复了3510指数行情的bug
* 数据库相关0.0.1 — 2022.08.01
>1. mysql化：增加将wind分钟数据mat文件转存至mysql，米筐分钟数据h5文件转存至mysql
>2. clickhouse化：增加将mysql转存至clickhouse，增加将米筐数据h5文件转存至clickhouse
* v2.6.3 — 2022.07.31 
>1. 修复了分钟数据计算因子类pure_fall_frequent和pure_fall_flexible的更新bug
>2. 修复了更新3510指数行情数据的bug，以及读取bug
>3. 将STATES中的默认参数startdate从20100101修改为20130101