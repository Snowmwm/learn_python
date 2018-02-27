import pandas as pd
import numpy as np
import datetime
import math
import talib
import random

def init(context): 
    #全局变量
    context.value = 0  
    context.stocks = []
    
    #定时器，每个月最后一个交易日开盘前运行一次choose
    scheduler.run_monthly(choose, tradingday=-1, time_rule='before_trading')
    
    #每个月最后一个交易日收盘前10分钟运行一次select
    scheduler.run_monthly(select, tradingday=-1, time_rule=market_close(minute=10))

    #每个月最后一个交易日收盘前8分钟运行一次trade
    scheduler.run_monthly(trade, tradingday=-1, time_rule=market_close(minute=8))
    
    #每天运行一次stoploss
    scheduler.run_daily(stoploss)
    
def handle_bar(context, bar_dict):
    pass 

def choose(context, bar_dict): 
    """在每月最后一个交易日开盘前运行，根据条件1-5选出满足要求的股票池"""
    
    #获取最近1年的日线收盘价数据
    al = all_instruments(type='CS')['order_book_id'].values.tolist() #股票列表
    end = context.now - datetime.timedelta(days=1)
    start = context.now - datetime.timedelta(days=366)
    df = get_price(al, start_date=start, end_date=end, fields='close')
    df = df.dropna(axis=1, how='any') #按列丢弃有缺失值的数据   
    
    #条件1：当前价格排名前25%
    series = df.ix[-1,:].sort_values(ascending=False) #取最后一天收盘价降序排序
    series = series.head(int(len(series)*0.25)+1) #取前25%
    #满足条件1的股票池
    stocks_1 = series.index.values
    
    
    #条件2：近一年涨幅排名前25%
    series = (df.ix[-1,:]/df.ix[1,:]).sort_values(ascending=False)
    series = series.head(int(len(series)*0.25)+1)
    #满足条件2的股票池
    stocks_2 = series.index.values

    
    #条件3：年度财务报表ROE>15%
    roe_panel = get_fundamentals(
        query(                              
            fundamentals.financial_indicator.adjusted_return_on_equity_average,
        )
        .filter(
            fundamentals.financial_indicator.adjusted_return_on_equity_average > 0.15
        ), interval='1y'
    )
    #满足条件3的股票池
    stocks_3 = roe_panel.columns.values 
    #stocks_3 = roe_panel['adjusted_return_on_equity_average'].columns.values 
    
    
    #条件4：季度财务报表EPS比前一季度增幅>18%
    eps_panel_1 = get_fundamentals(
        query(
            fundamentals.financial_indicator.earnings_per_share
        ).filter(
            fundamentals.financial_indicator.earnings_per_share > 0
        ),interval='2q'
    )
    eps = eps_panel_1['earnings_per_share'].T
    eps = eps.dropna(axis=0,how='any') #按行丢弃有缺失值的数据   
    #满足条件4的股票池
    stocks_4 = eps[eps.ix[:,0] > 1.18*eps.ix[:,1]].T.columns.values
    
    
    #条件5_1：年度财务报表EPS连续2年保持增长
    eps_panel_2 = get_fundamentals(
        query(
            fundamentals.financial_indicator.earnings_per_share
        ), interval='3y'
    )
    eps = eps_panel_2['earnings_per_share'].T
    eps = eps.dropna(axis=0,how='any')  
    #满足条件5_1的股票池
    stocks_5_1 = eps[(eps.ix[:,0] > eps.ix[:,1]) & (eps.ix[:,1] > \
        eps.ix[:,2])].T.columns.values
    
    #条件5_2: 当季财务报表EPS>100%
    eps_panel_3 = get_fundamentals(
        query(
            fundamentals.financial_indicator.earnings_per_share
        ).filter(
            fundamentals.financial_indicator.earnings_per_share > 1
        ),interval='1q'
    )
    #满足条件5_2的股票池
    stocks_5_2 = eps_panel_3.columns.values
    #stocks_5_2 = eps_panel_3['earnings_per_share'].columns.values
    
    #5_1、5_2取并集，然后1——5取交集
    stocks = set(stocks_1).intersection(set(stocks_2)).intersection(set( \
        stocks_3)).intersection(set(stocks_4)).intersection(set(stocks_5_1) \
        .union(set(stocks_5_2)))

    context.stocks = list(stocks)

def filterStk(stk, bar_dict, context):
    """过滤涨跌停股票"""
    yesterday = history_bars(stk, 2,'1d', 'close')[-1]
    zt = round(1.10 * yesterday,2)
    dt = round(0.97 * yesterday,2)
    if dt < bar_dict[stk].close < zt:
        return True
    else: 
        return False
        
def select(context, bar_dict):
    """在每月最后一个交易日收盘前10分钟运行
    过滤掉context.stocks中的涨跌停股票
    然后根据条件6、7进一步筛选股票"""
    
    stockList = context.stocks
    #过滤涨跌停股票
    #stockList = [stk for stk in stockList if filterStk(stk,bar_dict,context)]
    context.stocks = []
    #对于stockList中的每只股票
    for stock in stockList:
        price = history_bars(stock, 30, '1d','close') #取前30日收盘价
        volume = history_bars(stock, 30, '1d','volume') #取前30日成交量
    
        #如果满足条件6、7，将其放入context.stocks
        #MA5 > 1.1*MA25[5] 实际上等价于 MA5 > 1.082*MA30
        if (talib.MA(price, 5)[-1] > 1.082*talib.MA(price, 30)[-1]) and \
            (talib.MA(volume, 5)[-1] > 1.082*talib.MA(volume, 30)[-1]):
            context.stocks.append(stock)

    #如果选出的股票数大于10，从中随机选择10支股票
    if len(context.stocks) > 10:
        context.stocks = random.sample(context.stocks, 10)
        
        
def trade(context, bar_dict):
    #对于当前持有的所有股票,如果该股票不是context.stocks中的股票,卖出该股票
    for stock in context.portfolio.positions:
        if stock not in context.stocks:
            order_target_percent(stock, 0)
            
    #每只股票的权重
    if len(context.stocks) < 3:
        weight = 0  #如果选出的股票数小于3，则当月不交易
    else:
        weight = 1/len(context.stocks)
    
    #对于context.stocks中的每只股票，买入或卖出至总资金x权重
    for stock in context.stocks:
        order_target_percent(stock, weight)  
    #记录投资组合的总价值
    context.value = context.portfolio.market_value 
        
        
def stoploss(context,bar_dict):
    """回撤8%止损"""
    #回撤幅度，加0.0001是为了防止分母为0
    drawdown = 1 - context.portfolio.market_value/(context.value+0.0001) 
    if drawdown > 0.08:
        for stock in context.portfolio.positions:
            order_target_percent(stock,0)
