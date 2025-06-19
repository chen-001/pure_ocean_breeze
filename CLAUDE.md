# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述
这是一个量化多因子研究框架，专注于股票因子测试和分析。主要模块包括：
- `initialize`: 初始化配置和数据路径设置
- `jason.data`: 数据读取、处理和数据库操作
- `jason.labor`: 数据加工、回测和评价
- `jason.state`: 配置管理和装饰器
- `jason.withs`: 依赖管理

## 开发环境
- Python解释器: `/home/chenzongwei/.conda/envs/chenzongwei311/bin/python`
- Pip包管理: `/home/chenzongwei/.conda/envs/chenzongwei311/bin/pip`

## 核心命令

### 包构建与发布
```bash
# 构建包
/home/chenzongwei/.conda/envs/chenzongwei311/bin/python setup.py sdist bdist_wheel

# 安装本地开发版本
/home/chenzongwei/.conda/envs/chenzongwei311/bin/pip install -e .

# 测试包导入
/home/chenzongwei/.conda/envs/chenzongwei311/bin/python -c "import pure_ocean_breeze; print('包可以正常导入')"
```

### 文档构建
```bash
# 启动文档服务器
mkdocs serve

# 构建文档
mkdocs build
```

## 架构设计

### 模块结构
- `jason/data/`: 数据层
  - `read_data.py`: 读取日频、市场数据
  - `tools.py`: 数据处理工具函数，支持lru_cache优化
  - `database.py`: 数据库操作
  - `dicts.py`: 数据字典定义
- `jason/labor/`: 计算层
  - `process.py`: 数据加工和回测主模块
  - `comment.py`: 评价和注释功能
- `jason/state/`: 状态管理层
  - `homeplace.py`: 主配置类HomePlace
  - `decorators.py`: 装饰器工具
  - `states.py`: 状态管理

### 核心依赖
- 数据处理: pandas, polars, numpy, scipy
- 可视化: altair, plotly (避免使用matplotlib)
- 数据库: psycopg2, pyarrow
- 加速计算: rust_pyfunc, pandarallel
- 缓存优化: cachier, lru_cache

### 初始化流程
使用`pure_ocean_breeze.initialize.ini()`进行首次设置，配置：
- 日频数据路径
- 因子数据路径  
- barra数据路径
- 更新数据路径

配置信息会保存到用户目录下的pickle文件中。

## 开发注意事项
- 项目无单元测试框架，需要手动验证功能
- 使用Altair和Plotly进行数据可视化
- 代码中集成了多进程优化(pandarallel)和Rust加速(rust_pyfunc)
- 支持Jupyter Notebook环境检测
- 通过装饰器`@do_on_dfs`支持DataFrame操作的统一处理