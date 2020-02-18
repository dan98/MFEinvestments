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
# import local libraries

plt.close('all')

# constants
mu = 0.06
sigma = 0.20
years = 10
days = 365
n = years * days
dt = 1/365
s0 = 1.0

PLOT = False

D = 365
W = 52
M = 12

if __name__ == '__main__':
    
# =============================================================================
# Exercise 1  
# =============================================================================
    
    print('--------------\nExercise 1\n--------------')
    
    # compute path of share price
    dz = np.random.randn(n) * np.sqrt(dt)
    s = s0 * np.exp(np.cumsum((mu - 0.5*sigma*sigma)*dt + sigma*dz))
    
    # plot path of share price
    if PLOT:
        plt.figure()
        plt.plot(s)
        plt.legend(['Share price'])
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
        plt.grid('on')
    
    # annualized mean
    print('Annualized mean of log-returns: {}'.format(np.mean(log_returns)*D))
    # annualized standard deviation
    print('Annualized std of log-returns: {}'.format(np.std(log_returns)*np.sqrt(D)))
    
# =============================================================================
# Exercise 2
# =============================================================================
    
    print('\n--------------\nExercise 2\n--------------')
    
    np.random.seed(30)
    
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

    # compute monthly average share price
    s_monthly = s_daily.resample('M').mean()
    
    # plot monthly average share prices
    if PLOT:
        plt.figure()
        ax = s_monthly.plot(grid = True)
        ax.legend(['Monthly share prices'])

# =============================================================================
# Exercise 3 
# =============================================================================
    
    print('\n--------------\nExercise 3\n--------------')
    
    # compute simple return
    log_returns_daily = s_daily.pct_change()
    log_returns_daily = log_returns_daily.apply(lambda x: np.log(1+x))
    # weekly log returns
    s_weekly = s_daily.resample('W').first()
    log_returns_weekly = s_weekly.pct_change()
    log_returns_weekly = log_returns_weekly.apply(lambda x: np.log(1+x))
    # monthly log returns
    s_monthly = s_daily.resample('M').first()
    log_returns_monthly = s_monthly.pct_change()
    log_returns_monthly = log_returns_monthly.apply(lambda x: np.log(1+x))
    
    # print summary statistics
    print('Summary statistics for daily log returns')
    print(log_returns_daily.describe())
    print('\nSummary statistics for weekly log returns')
    print(log_returns_weekly.describe())
    print('\nSummary statistics for monthly log returns')
    print(log_returns_monthly.describe())

    # compute annualized mean of log-returns
    annualized_mean_daily = np.mean(log_returns_daily)*D
    print('\nAnnualized mean of daily log-returns: {}'.format(annualized_mean_daily))
    annualized_mean_weekly = np.mean(log_returns_weekly)*W
    print('Annualized mean of weekly log-returns: {}'.format(annualized_mean_weekly))
    annualized_mean_monthly = np.mean(log_returns_monthly)*M
    print('Annualized mean of mohthly log-returns: {}'.format(annualized_mean_monthly))
    
    # compute standard deviation of log-returns   
    annualized_std_daily = np.std(log_returns_daily)*np.sqrt(D)
    print('\nAnnualized std of daily log-returns: {}'.format(annualized_std_daily))
    annualized_std_weekly = np.std(log_returns_weekly)*np.sqrt(W)
    print('Annualized std of weekly log-returns: {}'.format(annualized_std_weekly))    
    annualized_std_monthly = np.std(log_returns_monthly)*np.sqrt(M)
    print('Annualized std of monthly log-returns: {}'.format(annualized_std_monthly))   
        
    # compute rolling one-year time series of std annualized data
    rolling_annualized_mean_daily = log_returns_daily.rolling(D).mean()*D
    rolling_annualized_mean_weekly = log_returns_weekly.rolling(W).mean()*W
    rolling_annualized_mean_monthly = log_returns_monthly.rolling(M).mean()*M
    
    # compute rolling one-year time series of mean annualized data
    rolling_annualized_std_daily = log_returns_daily.rolling(D).std()*np.sqrt(D)
    rolling_annualized_std_weekly = log_returns_weekly.rolling(W).std()*np.sqrt(W)
    rolling_annualized_std_monthly = log_returns_monthly.rolling(M).std()*np.sqrt(M)
    
    # plot
    if PLOT:
        plt.figure()
        rolling_annualized_mean_daily.plot(label = 'daily', title = 'Rolling annualized mean')
        rolling_annualized_mean_weekly.plot(label = 'weekly')
        rolling_annualized_mean_monthly.plot(label = 'monthly', grid = True)
        plt.legend()
    
        plt.figure()
        rolling_annualized_std_daily.plot(label = 'daily', title = 'Rolling annualized std')
        rolling_annualized_std_weekly.plot(label = 'weekly')
        rolling_annualized_std_monthly.plot(label = 'monthly', grid = True)
        plt.legend()
    
    # resample for bins of 365 days of daily log-returns
    mean_log_returns_daily = log_returns_daily.resample('A').mean()*D
    var_log_returns_daily = np.square(log_returns_daily).resample('A').mean()*D
    
    # resample for bins of 365 days of monthly log-returns
    mean_log_returns_monthly = log_returns_monthly.resample('A').mean()*M
    var_log_returns_monthly = np.square(log_returns_monthly).resample('A').mean()*M
    
    # print results
    print('\nMean of mean estimator (daily): {}'.format(mean_log_returns_daily.mean()))
    print('Variance of mean estimator (daily): {}'.format(mean_log_returns_daily.var()))
    
    print('\nMean of variance estimator (daily): {}'.format(var_log_returns_daily.mean()))
    print('Variance of variance estimator (daily): {}.'.format(var_log_returns_daily.var()))
    
    print('\nMean of mean estimator (monthly): {}'.format(mean_log_returns_monthly.mean()))
    print('Variance of mean estimator (monthly): {}'.format(mean_log_returns_monthly.var()))
    
    print('\nMean of variance estimator (monthly): {}'.format(var_log_returns_monthly.mean()))
    print('Variance of variance estimator (monthly): {}.'.format(var_log_returns_monthly.var()))

# =============================================================================
# Exercise 4
# =============================================================================

    print('\n--------------\nExercise 4\n--------------')
    
    
    
    