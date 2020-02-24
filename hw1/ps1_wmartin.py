#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:00:41 2020

@author: williammartin
"""

# import standard libraries
import datetime
import numpy as np
# import third-party libraries
import matplotlib.pyplot as plt
import pandas as pd
import wrds
# import local libraries

plt.close('all')

# constants
mu = 0.06
sigma = 0.20
years = 10
days = 365
n = years * days
dt = 1/365
s0 = 100

PLOT = True

D = 365
W = 52
M = 12

np.random.seed(100)

if __name__ == '__main__':
    
# =============================================================================
# Exercise 1  
# =============================================================================
    
    print('--------------\nExercise 1\n--------------')
    
    # compute path of share price
    dz1 = np.random.randn(n) * np.sqrt(dt)
    # dz = np.insert(dz, 0, 0, axis = 0)
    s = s0 * np.exp(np.cumsum((mu - 0.5*sigma*sigma)*dt + sigma*dz1))
    
    # plot path of share price
    if PLOT:
        plt.figure()
        plt.plot(s)
        plt.legend(['Share price'])
        plt.xlabel('Observations')
        plt.ylabel('Price')
        plt.grid('on')

    # simple return
    simple_returns = np.diff(s)/s[0:-1]
    # continuously compounded returns
    log_returns = [np.log(1 + r) for r in simple_returns]
        
    # plot log returns
    if PLOT:
        plt.figure()
        plt.plot(log_returns)
        plt.legend(['Daily log returns'])
        plt.xlabel('Observations')
        plt.ylabel('Log return')
        plt.grid('on')
    
    # annualized mean
    print('Annualized mean of log-returns: {}'.format(np.mean(log_returns)*D))
    # annualized standard deviation
    print('Annualized std of log-returns: {}'.format(np.std(log_returns)*np.sqrt(D)))
    
# =============================================================================
# Exercise 2
# =============================================================================
    
    print('\n--------------\nExercise 2\n--------------')
    
    # daily share prices
    start = datetime.datetime(1950, 1, 1)
    end = datetime.datetime(2018, 12, 31)
    index = pd.period_range(start, end, freq = 'D')
    n = len(index)
    dz = np.random.randn(n) * np.sqrt(dt)
    s_daily = s0 * np.exp(np.cumsum((mu - 0.5*sigma*sigma)*dt + sigma*dz))
    s_daily = pd.Series(s_daily, index)
    
    # number of observations
    print('Number of observations: {}'.format(n))
    
    # plot daily share prices
    if PLOT:
        plt.figure()
        ax = s_daily.plot(grid = True)
        ax.legend(['Daily share prices'])
        plt.xlabel('Time')
        plt.ylabel('Price')

    # compute monthly average share price
    s_monthly = s_daily.resample('M').mean()
    
    # plot monthly average share prices
    if PLOT:
        plt.figure()
        ax = s_monthly.plot(grid = True)
        ax.legend(['Monthly average share prices'])
        plt.xlabel('Time')
        plt.ylabel('Price')

# =============================================================================
# Exercise 3 
# =============================================================================
    
    print('\n--------------\nExercise 3\n--------------')
    
    start = datetime.datetime(1950, 1, 1)
    end = datetime.datetime(2018, 12, 31)
    index = pd.period_range(start, end, freq = 'D')
    n = len(index)
    dz = np.random.randn(n) * np.sqrt(dt)
    s_daily = s0 * np.exp(np.cumsum((mu - 0.5*sigma*sigma)*dt + sigma*dz))
    s_daily = pd.Series(s_daily, index)
    
    def run_analysis(data, label, d = D):
    
        # compute simple return
        if label == 'simulated':
            log_returns_daily = data.pct_change()
        else:
            log_returns_daily = data
        log_returns_daily = log_returns_daily.apply(lambda x: np.log(1+x))
        # weekly log returns
        if label == 'simulated':
            s_weekly = data.resample('W').last()
            log_returns_weekly = s_weekly.pct_change()
        else:
            s_weekly = data.resample('W').sum()
            log_returns_weekly = s_weekly
        log_returns_weekly = log_returns_weekly.apply(lambda x: np.log(1+x))
        # monthly log returns
        if label == 'simulated':
            s_monthly = data.resample('M').last()
            log_returns_monthly = s_monthly.pct_change()
        else:
            s_monthly = data.resample('M').sum()
            log_returns_monthly = s_monthly
        log_returns_monthly = log_returns_monthly.apply(lambda x: np.log(1+x))
        
        # print summary statistics
        print('Summary statistics for daily log returns')
        print(log_returns_daily.describe())
        print('\nSummary statistics for weekly log returns')
        print(log_returns_weekly.describe())
        print('\nSummary statistics for monthly log returns')
        print(log_returns_monthly.describe())
    
        # compute annualized mean of log-returns
        annualized_mean_daily = np.mean(log_returns_daily)*d
        print('\nAnnualized mean of daily log-returns: {}'.format(annualized_mean_daily))
        annualized_mean_weekly = np.mean(log_returns_weekly)*W
        print('Annualized mean of weekly log-returns: {}'.format(annualized_mean_weekly))
        annualized_mean_monthly = np.mean(log_returns_monthly)*M
        print('Annualized mean of mohthly log-returns: {}'.format(annualized_mean_monthly))
        
        # compute standard deviation of log-returns   
        annualized_std_daily = np.std(log_returns_daily)*np.sqrt(d)
        print('\nAnnualized std of daily log-returns: {}'.format(annualized_std_daily))
        annualized_std_weekly = np.std(log_returns_weekly)*np.sqrt(W)
        print('Annualized std of weekly log-returns: {}'.format(annualized_std_weekly))    
        annualized_std_monthly = np.std(log_returns_monthly)*np.sqrt(M)
        print('Annualized std of monthly log-returns: {}'.format(annualized_std_monthly))   
            
        # compute rolling one-year time series of std annualized data
        rolling_annualized_mean_daily = log_returns_daily.rolling(d).mean()*d
        rolling_annualized_mean_weekly = log_returns_weekly.rolling(W).mean()*W
        rolling_annualized_mean_monthly = log_returns_monthly.rolling(M).mean()*M
        
        # compute rolling one-year time series of mean annualized data
        rolling_annualized_std_daily = log_returns_daily.rolling(d).std()*np.sqrt(d)
        rolling_annualized_std_weekly = log_returns_weekly.rolling(W).std()*np.sqrt(W)
        rolling_annualized_std_monthly = log_returns_monthly.rolling(M).std()*np.sqrt(M)
        
        # plot
        if PLOT:
            plt.figure()
            rolling_annualized_mean_daily.plot(label = 'daily', title = 'Annualized mean of log-returns using a rolling one-year window ({})'.format(label))
            rolling_annualized_mean_weekly.plot(label = 'weekly')
            rolling_annualized_mean_monthly.plot(label = 'monthly')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Annualized mean of log-returns')
        
            plt.figure()
            rolling_annualized_std_daily.plot(label = 'daily', title = 'Annualized std of log-returns using a rolling one-year window ({})'.format(label))
            rolling_annualized_std_weekly.plot(label = 'weekly')
            rolling_annualized_std_monthly.plot(label = 'monthly')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Annualized std deviation of log-returns')
        
        if label == 'simulated':
            # resample for bins of 365 days of daily log-returns
            mean_log_returns_daily = log_returns_daily.resample('A').mean()*d
            var_log_returns_daily = np.square(log_returns_daily).resample('A').mean()*d
            
            # resample for bins of 365 days of monthly log-returns
            mean_log_returns_monthly = log_returns_monthly.resample('A').mean()*M
            var_log_returns_monthly = np.square(log_returns_monthly).resample('A').mean()*M
            
            # print results
            print('\nMean of mean estimator (daily): {}'.format(mean_log_returns_daily.mean()))
            print('Theoretical value: {}'.format(mu))
            print('Variance of mean estimator (daily): {}'.format(mean_log_returns_daily.var()))
            print('Theoretical value: {}'.format(0.2*0.2))
    
            print('\nMean of mean estimator (monthly): {}'.format(mean_log_returns_monthly.mean()))
            print('Theoretical value: {}'.format(mu))
            print('Variance of mean estimator (monthly): {}'.format(mean_log_returns_monthly.var()))
            print('Theoretical value: {}'.format(0.2*0.2))
            
            print('\nMean of variance estimator (daily): {}'.format(var_log_returns_daily.mean()))
            mean_variance_daily = sigma**2 + mu**2/365
            print('Theoretical value: {}'.format(mean_variance_daily))
            print('Variance of variance estimator (daily): {}.'.format(var_log_returns_daily.var()))
            variance_variance_daily = 2*sigma**4/365 + 4*mu**2*sigma**2/(365**2)
            print('Theoretical value: {}'.format(variance_variance_daily))
            
            
            print('\nMean of variance estimator (monthly): {}'.format(var_log_returns_monthly.mean()))
            mean_variance_monthly = sigma**2 + mu**2/12
            print('Theoretical value: {}'.format(mean_variance_monthly))
            print('Variance of variance estimator (monthly): {}.'.format(var_log_returns_monthly.var()))
            variance_variance_monthly = 2*sigma**4/12 + 4*mu**2*sigma**2/(12**2)
            print('Theoretical value: {}'.format(variance_variance_monthly))

    # run exercise 3 with simulated data
    run_analysis(data = s_daily, label = 'simulated')

# =============================================================================
# Exercise 4
# =============================================================================

    print('\n--------------\nExercise 4\n--------------')
    
    # connect to databse and download csv files (run once)
    # db = wrds.Connection(wrds_username = 'wmartin')
    # db.create_pgpass_file() # run once
    """
    aapl = db.raw_sql("select date, ret from crsp.dsf where permco in (7) and date>='2001-01-01' and date<='2018-12-31'")
    gs = db.raw_sql("select date, ret from crsp.dsf where permco in (35048) and date>='2001-01-01' and date<='2018-12-31'")
    msft = db.raw_sql("select date, ret from crsp.dsf where permco in (8048) and date>='2001-01-01' and date<='2018-12-31'")
    pg = db.raw_sql("select date, ret from crsp.dsf where permco in (21446) and date>='2001-01-01' and date<='2018-12-31'")
    ge = db.raw_sql("select date, ret from crsp.dsf where permco in (20792) and date>='2001-01-01' and date<='2018-12-31'")
    aapl.to_csv('aapl.csv')
    gs.to_csv('gs.csv')
    msft.to_csv('msft.csv')
    pg.to_csv('pg.csv')
    ge.to_csv('ge.csv')
    """
    start = datetime.datetime(2001, 1, 1)
    end = datetime.datetime(2018, 12, 31)
    
    # read csvs
    use_cols = ['ret', 'date']
    index_col = 'date'
    
    aapl = pd.read_csv('aapl.csv', usecols = use_cols, index_col = index_col)
    gs = pd.read_csv('gs.csv', usecols = use_cols, index_col = index_col)
    msft = pd.read_csv('msft.csv', usecols = use_cols, index_col = index_col)
    pg = pd.read_csv('pg.csv', usecols = use_cols, index_col = index_col)
    ge = pd.read_csv('ge.csv', usecols = use_cols, index_col = index_col)

    # aggregate data in dictionary
    data = dict(aapl = aapl, gs = gs, msft = msft, pg = pg, ge = ge)
    
    for company, df in data.items():
        print('\nRunning analysis for {}'.format(company))
        
        # index to datetime
        df.index = pd.to_datetime(df.index)
        series = df['ret']
        
        # run analysis on series
        run_analysis(series, company, d = 252)
    
    
    
    
    
    
    
    
    
    
    