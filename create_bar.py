import kungfu
from kungfu.wingchun.constants import *
import kungfu.yijinjing.time as kft
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Mapping, Text, Sequence, Iterable, Dict ,List,Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass


lf = kungfu.__binding__.longfist

SOURCE = "custom"
EXCHANGE = Exchange.BINANCE_USD_FUTURE
INTERVAL = 15 * kft.NANO_PER_MINUTE
instrument_ids =  ['1000SHIBUSDT', 'ATAUSDT', 'EOSUSDT', 'TRBUSDT', '1INCHUSDT', 'ICXUSDT', 'OGNUSDT', 'TRXUSDT', 'AAVEUSDT' , 'BTCUSDT','ETHUSDT','AVAXUSDT', 'DASHUSDT',
                                    'IOSTUSDT', 'LRCUSDT' , 'OMGUSDT', 'SANDUSDT', 'ADAUSDT', 'C98USDT', 'DEFIUSDT', 'FILUSDT', 'IOTAUSDT', 'LTCUSDT', 'ONEUSDT', 'SFPUSDT', 'ZILUSDT',
                                    'ALGOUSDT', 'BAKEUSDT', 'CELOUSDT', 'FLMUSDT', 'IOTXUSDT', 'MANAUSDT', 'ONTUSDT', 'SKLUSDT', 'VETUSDT', 'ZRXUSDT', 'ALICEUSDT', 'CELRUSDT', 'FTMUSDT',
                                    'KAVAUSDT', 'MASKUSDT', 'PEOPLEUSDT', 'SOLUSDT', 'WAVESUSDT' ,'ALPHAUSDT', 'BANDUSDT', 'CHRUSDT', 'DOGEUSDT', 'GALAUSDT', 'KLAYUSDT', 'MATICUSDT',
                                    'QTUMUSDT', 'STMXUSDT', 'XEMUSDT', 'ANKRUSDT', 'BATUSDT', 'CHZUSDT', 'DOTUSDT', 'GRTUSDT', 'MKRUSDT', 'REEFUSDT', 'STORJUSDT', 'XLMUSDT','BCHUSDT',
                                    'COMPUSDT', 'DUSKUSDT', 'NEARUSDT', 'SUSHIUSDT', 'ARPAUSDT', 'BELUSDT', 'COTIUSDT', 'HBARUSDT', 'LINAUSDT', 'NEOUSDT', 'ROSEUSDT', 'SXPUSDT', 'XRPUSDT',
                                    'BLZUSDT', 'LINKUSDT', 'RSRUSDT']


class Bar:
    def __init__(self):
        self.volume:int = 0
        self.open:int = 0
        self.close:int = 0
        self.high:int = 0
        self.low:int = 0
        self.last_volume:int = 0
        self.bar_time_interval:int = 60 * kft.NANO_PER_SECOND # 每条bar线的间隔
        self.size:int = 100 # bar线的窗口大小
        self.open_series:pd.Series = pd.Series([0]*self.size)
        self.close_series:pd.Series = pd.Series([0]*self.size)
        self.high_series:pd.Series = pd.Series([0]*self.size)
        self.low_series:pd.Series = pd.Series([0]*self.size)
        self.volume_series:pd.Series = pd.Series([0]*self.size)
        
    def new_bar(self):
        self.last_price = self.transaction.price
        volume = self.transaction.volume
        self.open = self.last_price
        self.close = self.last_price
        self.high = self.last_price
        self.low = self.last_price
        self.volume = volume
        
    def data_update(self,transaction):  # ochlv数据更新
        if self.volume == 0: # 如果该合约从来没有发生过交易
            self.transaction = transaction
            self.new_bar()
        else:
            volume = transaction.volume
            self.last_price = transaction.price
            self.close = self.last_price 
            self.high = max(self.high, self.last_price)
            self.low = min(self.low, self.last_price)
            self.volume += volume
    
    def bar_update(self,context,event): # bar数据更新
        volume = self.volume - self.last_volume

        self.open_series[:-1] = self.open_series[1:]
        self.high_series[:-1] = self.high_series[1:]
        self.low_series[:-1] = self.low_series[1:]
        self.close_series[:-1] = self.close_series[1:]
        self.volume_series[:-1] = self.volume_series[1:]

        self.open_series[self.size - 1] = self.open
        self.high_series[self.size - 1] = self.high
        self.low_series[self.size - 1] = self.low
        self.close_series[self.size - 1] = self.close
        self.volume_series[self.size - 1] = volume
        
        self.last_volume = self.volume
        self.open = self.close
        self.high = self.close
        self.low = self.close        

def pre_start(context):  
    context.log.info('k_line_start')    
    context.subscribe(SOURCE, instrument_ids, EXCHANGE)
    context.instruments_bar_dict = defaultdict(float) # 用于存储Bar数据
    context.GOLDEN_DEATH_SIGNAL = defaultdict(float)
    for id_ in instrument_ids:
        context.instruments_bar_dict[f'{id_}.{EXCHANGE}'] = Bar()
        context.add_time_interval(Bar().bar_time_interval, context.instruments_bar_dict[f'{id_}.{EXCHANGE}'].bar_update)  # k线间隔在这里触发
    
    context.add_time_interval(INTERVAL, on_bar)

# 关于k线的因子在此计算
def on_bar(context, event):
    for key,values in context.instruments_bar_dict.items():
        # 长短均线
        long_ma = values.close_series.rolling(30).mean()
        short_ma = values.close_series.rolling(15).mean()
        # 长短均线差
        last_diff = long_ma.iloc[-1] - short_ma.iloc[-1]
        second_last_diff = long_ma.iloc[-2] - short_ma.iloc[-2]
        # 按照均线差的变动程度作为因子
        if second_last_diff<0 and last_diff>0: # 死叉
            context.GOLDEN_DEATH_SIGNAL[key] = -1 * pct_change_(second_last_diff,last_diff)
        elif second_last_diff>0 and last_diff<0: # 金叉
            context.GOLDEN_DEATH_SIGNAL[key] = pct_change_(second_last_diff,last_diff)
        else:
            context.GOLDEN_DEATH_SIGNAL[key] = 0
    context.log.info(f'{kft.to_datetime(context.now())}') # 确保系统在运行
    context.publish_synthetic_data(
    "GOLDEN_DEATH_SIGNAL",
    json.dumps(context.GOLDEN_DEATH_SIGNAL),
    )
    context.GOLDEN_DEATH_SIGNAL.clear()

def on_transaction(context, transaction: lf.types.Transaction, location, dest):
    context.instruments_bar_dict[f'{transaction.instrument_id}.{transaction.exchange_id}'].data_update(transaction)

def pct_change_(second_last_diff,last_diff):
    try:
        return abs(last_diff/second_last_diff)
    except:
        return 0

