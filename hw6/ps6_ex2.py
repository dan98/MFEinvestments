#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:50:57 2020

@author: williammartin
"""

# import standard libraries
import numpy as np
# import third-party libraries
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import wrds
# import local libraries

plt.close('all')
DOWNLOAD = False

class Stock:
    
    def __init__(self, permno, df):
        self.permno = permno
        self.df = df
        
    def computeExcess(self, rf):
        self.df['excess'] = self.df['ret'].sub(rf['rf'])
        
    def computeBeta(self, market, start = '2000-01-01', end = '2019-12-31'):
        self.covariance = self.df['excess'][start:end].cov(market['excess'][start:end])
        self.variance = market['excess'][start:end].var()
        self.beta = self.covariance/self.variance
        
class Portfolio(Stock):
    
    def __init__(self, bucket, df, start = '2000-01-01', end = '2019-12-31'):
        self.bucket = bucket
        self.df = df
        self.meanret = self.df['ret'][start:end].mean()
        

if __name__ == '__main__':
    
    if DOWNLOAD:
        
        start = '2000-01-01'
        end = '2019-12-31'
        
        db = wrds.Connection(wrds_username = 'wmartin')
        # db.create_pgpass_file()
        
        # get risk free
        rf = db.raw_sql("select mcaldt,tmytm "
                        "from crsp.tfz_mth_rf "           
                        "where kytreasnox = 2000001 "
                        "and mcaldt>='{}'"
                        "and mcaldt<='{}'".format(start, end), 
                        date_cols = ['mcaldt'])
        
        # get crsp value-weighted idnex return
        crsp_index = db.raw_sql("select date,vwretd "
                                "from crsp.msi "
                                "where date>='{}'"
                                "and date<='{}'".format(start, end), 
                                date_cols = ['date'])
        
        # get crsp stock event
        crsp_stock = db.raw_sql("select a.permno, a.date, "
                                "b.shrcd, b.exchcd, a.ret "
                                "from crsp.msf as a "
                                "left join crsp.msenames as b " 
                                "on a.permno=b.permno " 
                                "and b.namedt<=a.date " 
                                "and a.date<=b.nameendt "
                                "where a.date between '{}' and '{}' "
                                "and b.exchcd in (1, 2) "
                                "and b.shrcd in (10, 11) ".format(start, end), 
                                date_cols=['date']) 
        
        crsp_stock = crsp_stock.drop(columns = ['shrcd', 'exchcd'])
        
        # write to csv raw
        rf.to_csv('rf.csv')
        crsp_index.to_csv('crsp_index.csv')
        crsp_stock.to_csv('crsp_stock.csv')

        
    else:
        
        # read locally
        rf = pd.read_csv('rf.csv', usecols = [1, 2])
        rf['mcaldt'] = pd.to_datetime(rf['mcaldt'])
        
        crsp_index = pd.read_csv('crsp_index.csv', usecols = [1, 2])
        crsp_index['date'] = pd.to_datetime(crsp_index['date'])
        
        crsp_stock = pd.read_csv('crsp_stock.csv', usecols = [1, 2, 3])
        crsp_stock['date'] = pd.to_datetime(crsp_stock['date'])
        
    
    # clean rf
    rf = rf.rename(columns = {'mcaldt': 'date', 'tmytm': 'rf'})
    
    # clean data a bit
    rf = rf.set_index('date')
    crsp_index = crsp_index.set_index('date')
    crsp_stock = crsp_stock.set_index('date')
    
    # floatify data
    crsp_stock['permno'] = \
        crsp_stock['permno'].astype(int)
        
    # correct rf to monhtly returns
    rf = rf.applymap(lambda x: np.exp(x/12/100) - 1)
    
    # split crsp_stock by stock
    crsp_stock_grouped = crsp_stock.groupby('permno')
    
    # keep stocks with 240 observations at least
    stocks = {}
    for permno, df in crsp_stock_grouped:
        if len(df) >= 240:
            df = df.drop(columns = ['permno'])
            df = df.sort_index()
            stocks[permno] = Stock(permno, df)
     
    # compute excess returns
    for _, s in stocks.items():
        s.computeExcess(rf)
    crsp_index['excess'] = crsp_index['vwretd'].sub(rf['rf'])

# =============================================================================
# Using full sample
# =============================================================================     
    # compute beta of stocks
    for _, s in stocks.items():
        s.computeBeta(crsp_index)
    
    # create dataframe of betas
    betas = {permno: s.beta for permno, s in stocks.items()}
    # put betas in bucket
    betas_bucket = 1+pd.qcut(pd.Series(betas), 10, labels = False)
    # save bucket in class objects
    for permno, b in betas_bucket.items():
        stocks[permno].bucket = b
        
    # compute equal-weighted average average return for each bucket for each month
    # each item in returns is a dataframe with column permno and index dates
    portfolios = {}
    for b in np.sort(betas_bucket.unique()):
        permnos = betas_bucket[betas_bucket == b].index
        returns = pd.concat([stocks[p].df['ret'] for p in permnos], axis = 1)
        returns.columns = permnos
        # compute equal-weighted average average return for the bucket
        ew_returns = pd.DataFrame(returns.mean(axis = 1))
        ew_returns = ew_returns.rename(columns = {0: 'ret'})
        # create portfolio object
        portfolio = Portfolio(b, ew_returns)
        # compute excess returns
        portfolio.computeExcess(rf)
        # compute beta
        portfolio.computeBeta(crsp_index)
        portfolios[b] = portfolio
    
    # plot average returns vs betas of each portfolio
    x = np.array([p.beta for _, p in portfolios.items()])
    x = x.reshape(-1, 1)
    y = np.array([p.meanret for _, p in portfolios.items()])
    
    # fit line on x and y
    regressor = LinearRegression()
    regressor = regressor.fit(x, y)
        
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.scatter(x, y, color = '#4f6ddb')
    
    # get limit of graphs and plot fitted line
    xx = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
    y_fit = regressor.intercept_ + regressor.coef_*xx
    ax.plot(xx, y_fit, color = '#1bb582')
    
    # compute CAPM line
    r0 = rf.mean().values[0]
    excess_mean = crsp_index['excess'].mean()
    y_capm = r0 + excess_mean*xx
    ax.plot(xx, y_capm, color = '#e07e07')
    
    plt.title('Returns vs. betas (full sample data)')
    ax.set_xlabel('beta')
    ax.set_ylabel('returns')
    ax.legend(['fitted line', 'Average market excess return', '10 average portfolio'])
    
    print('Market risk premium: {}'.format(excess_mean))
    print('Real slope: {}'.format(regressor.coef_[0]))
    
# =============================================================================
# Using out-of-sample data
# =============================================================================
    # compute beta of stocks
    for _, s in stocks.items():
        s.computeBeta(crsp_index, '2000-01-01', '2010-12-31')
        
    # create dataframe of betas
    betas = {permno: s.beta for permno, s in stocks.items()}
    # put betas in bucket
    betas_bucket = 1+pd.qcut(pd.Series(betas), 10, labels = False)
    # save bucket in class objects
    for permno, b in betas_bucket.items():
        stocks[permno].bucket = b
    
    # compute equal-weighted average average return for each bucket for each month
    # each item in returns is a dataframe with column permno and index dates
    portfolios2 = {}
    for b in np.sort(betas_bucket.unique()):
        permnos = betas_bucket[betas_bucket == b].index
        returns = pd.concat([stocks[p].df['ret'] for p in permnos], axis = 1)
        returns.columns = permnos
        # compute equal-weighted average average return for the bucket
        ew_returns = pd.DataFrame(returns.mean(axis = 1))
        ew_returns = ew_returns.rename(columns = {0: 'ret'})
        # create portfolio object
        portfolio = Portfolio(b, ew_returns, '2010-01-01', '2019-12-31')
        # compute excess returns
        portfolio.computeExcess(rf)
        # compute beta
        portfolio.computeBeta(crsp_index, '2000-01-01', '2010-12-31')
        portfolios2[b] = portfolio
      
    # plot average returns vs betas of each portfolio
    x = np.array([p.beta for _, p in portfolios2.items()])
    x = x.reshape(-1, 1)
    y = np.array([p.meanret for _, p in portfolios2.items()])
    
    # fit line on x and y
    regressor = LinearRegression()
    regressor = regressor.fit(x, y)
        
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.scatter(x, y, color = '#4f6ddb')
    
    # get limit of graphs and plot fitted line
    xx = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
    y_fit = regressor.intercept_ + regressor.coef_*xx
    ax.plot(xx, y_fit, color = '#1bb582')
    
    plt.title('Returns vs. betas (out-of-sample data)')
    ax.set_xlabel('beta')
    ax.set_ylabel('returns')
    ax.legend([ 'fitted line', '10 average portfolio'])
    
    print('\nReal slope: {}'.format(regressor.coef_[0]))
    
    # compute betas of the second sample.
    for _, portfolio in portfolios2.items():
        portfolio.computeBeta(crsp_index, '2010-01-01', '2019-12-31')   
        
    # get betas of second sample
    y = np.array([p.beta for _, p in portfolios2.items()])
    
    # plot betas is second sample against betas of first sample
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.scatter(x, y, color = '#4f6ddb')
    
    plt.title('Betas vs. betas')
    ax.set_xlabel('beta (first sample)')
    ax.set_ylabel('beta (seconds sample')
    
    
    
    
    
    
            
    
            
        