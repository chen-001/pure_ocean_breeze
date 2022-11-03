import time
import datetime
import os

import numpy as np
import pandas as pd
import pickledb
import psycopg2 as pg
import psycopg2.extras as extras
import pymysql
import requests
import scipy.io as scio
import tqdm
from cachier import cachier
from loguru import logger
from psycopg2.extensions import AsIs, register_adapter
from sqlalchemy import BIGINT, FLOAT, INT, VARCHAR, create_engine

try:
    import rqdatac

    rqdatac.init()
except Exception:
    print("暂时未连接米筐")

import knockknock as kk
import matplotlib.pyplot as plt

plt.style.use(["science", "no-latex", "notebook"])
plt.rcParams["axes.unicode_minus"] = False
import copy
import pickle

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib as mpl
import scipy.stats as ss
import statsmodels.formula.api as smf


import smtplib
import warnings
from collections import Iterable
from email.header import Header
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import lru_cache, partial, reduce
from typing import Callable, Union

import plotly.express as pe
import plotly.io as pio
from dateutil.relativedelta import relativedelta
from tenacity import retry

warnings.filterwarnings("ignore")
import bs4
from wrapt_timeout_decorator import timeout
import pyfinance.ols as po
from texttable import Texttable
import numpy_ext as npext
from xpinyin import Pinyin
import cufflinks as cf
cf.set_config_file(offline=True)
from plotly.tools import FigureFactory as FF
import plotly.graph_objects as go
import plotly.tools as plyoo