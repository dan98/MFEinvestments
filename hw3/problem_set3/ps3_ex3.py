#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:12:21 2020

@author: williammartin
"""

# import standard libraries
import copy
import matplotlib.pyplot as plt
import numpy as np
# import third-party libraries
import pandas as pd
import wrds
# import local libraries

plt.close('all')

DOWNLOAD = False

if __name__ == '__main__':
    
    if DOWNLOAD:
        # connect to databse and download csv files (run once)
        db = wrds.Connection(wrds_username = 'wmartin')
        db.create_pgpass_file() # run once
    
        vwretd = db.raw_sql("select date, vwretd from crsp.msi where date>='1960-01-01' and date<='2019-12-31'", date_cols=['date'])
        bonds = db.raw_sql("select caldt, b2ret from crsp.mcti where caldt>='1960-01-01' and caldt<='2019-12-31'", date_cols=['caldt'])
        tbill = db.raw_sql("select mcaldt, tmytm from crsp.tfz_mth_rf where kytreasnox = 2000001 and mcaldt>='1960-01-01' and mcaldt<='2019-12-31'")
        
        vwretd.to_csv('vwretd.csv')
        bonds.to_csv('bonds.csv')
        tbill.to_csv('tbill.csv')
    
    # read csv files    
    crsp = pd.read_csv('vwretd.csv', usecols = ['vwretd', 'date'], index_col = 'date')
    bond = pd.read_csv('bonds.csv', usecols = ['b2ret', 'caldt'], index_col = 'caldt')
    bill = pd.read_csv('tbill.csv', usecols = ['tmytm', 'mcaldt'], index_col = 'mcaldt')

    # index to datetime
    crsp.index = pd.to_datetime(bond.index)
    bond.index = pd.to_datetime(bond.index)
    bill.index = pd.to_datetime(bond.index)
    
    # transform t bills
    bill = bill.apply(lambda x: np.exp(x/(12*100))-1)
    
    # concatenate into data
    risky = pd.concat([crsp, bond], axis = 1)    
    free = copy.deepcopy(bill)
    series = pd.concat([risky, free], axis = 1)
    
    # compute excess
    excess_vwretd = series['vwretd'] - series['tmytm']
    excess_b2ret = series['b2ret'] - series['tmytm']
    excess = pd.concat([excess_vwretd, excess_b2ret], axis = 1)
    excess = excess.rename(columns = {0: 'vwretd', 1: 'b2ret'})
  
# =============================================================================
# First part
# =============================================================================
    
    # compute mean, std and correlation matrix
    print('Mean of returns')
    print(series.mean(axis = 0))
    
    print('\nStandard deviation of returns')    
    print(series.std(axis = 0))
    
    print('\nCorrelation matrix of returns')
    print(series.corr())
    
    # compute tangency portfolio
    SIGMA = series.cov()[['vwretd', 'b2ret']]
    SIGMA = SIGMA.loc[['vwretd', 'b2ret']]
    SIGMA_inv = np.linalg.inv(SIGMA) 
    mu = risky.mean().values
    ones = np.ones((2, 1))
    A = ones.T.dot(SIGMA_inv.dot(ones))
    B = ones.T.dot(SIGMA_inv.dot(mu))
    C = mu.T.dot(SIGMA_inv.dot(mu))    
    D = A*C - B**2    
    R0 = free.mean().values
    
    # compute tangency weights
    wtan = (SIGMA_inv.dot(mu - R0))/(B - A*R0)
    # compute metrics of tangency portfolio
    mean_wtan = wtan.dot(mu)
    std_wtan = np.sqrt(wtan.dot(SIGMA.dot(wtan.T)))
    sr_wtan = (mean_wtan - R0)/std_wtan
    
    print('\nTangency portfolio')
    print('{:.2f}% in wvretd and {:.2f}% in b2ret'.format(wtan[0][0], wtan[0][1]))
    print('Mean of tangency portfolio')
    print(mean_wtan[0])
    print('Standard deviation of tangency portfolio')
    print(std_wtan[0][0])
    print('Sharpe ratio of tangency portfolio')
    print(sr_wtan[0][0])
    
    # compute metrics for 60/40 portfolio
    wport = np.ones((1, 2))*[0.6, 0.4]
    # compute metrics of 60/40 portfolio
    mean_wport = wport.dot(mu)
    std_wport = np.sqrt(wport.dot(SIGMA.dot(wport.T)))
    sr_wport = (mean_wport - R0)/std_wport
    
    print('\n60/40 portfolio')
    print('{:.2f}% in wvretd and {:.2f}% in b2ret'.format(wport[0][0], wport[0][1]))
    print('Mean of 60/40 portfolio')
    print(mean_wport[0])
    print('Standard deviation of 60/40 portfolio')
    print(std_wport[0][0])
    print('Sharpe ratio of 60/40 portfolio')
    print(sr_wport[0][0])
    
# =============================================================================
# Second part
# =============================================================================    
    
    
    # according to appendix A of the paper
    # compute weights of RP unlevered portfolio
    excess_std = excess.std()
    excess_std_inv = excess_std.apply(lambda x: 1/x)
    k = 1/sum(excess_std_inv)
    wrpu = k*excess_std_inv
    wrpu = wrpu.values
    
    # compute metrics of RP-unlevered portfolio
    mean_wrpu = wrpu.dot(mu)
    std_wrpu = np.sqrt(wrpu.dot(SIGMA.dot(wrpu.T)))
    sr_wrpu = (mean_wrpu - R0)/std_wrpu
    
    print('\nRP-unlevered portfolio')
    print('{:.2f}% in wvretd and {:.2f}% in b2ret'.format(wrpu[0], wrpu[1]))
    print('Mean of RP-unlevered portfolio')
    print(mean_wrpu)
    print('Standard deviation of RP-unlevered portfolio')
    print(std_wrpu)
    print('Sharpe ratio of RP-unlevered portfolio')
    print(sr_wrpu[0])
    
    
    # compute weights of RP levered portofolio
    k = std_wport / std_wrpu
    wrpl = k*wrpu    
    wrpl = wrpl[0]
    wfree = 1 - sum(wrpl)
    
    # compute metrics of RP-levered portfolio
    mean_wrpl = wrpl.dot(mu) + wfree*R0
    std_wrpl = np.sqrt(wrpl.dot(SIGMA.dot(wrpl.T)))
    sr_wrpl = (mean_wrpl - R0)/std_wrpl

    print('\nRP-levered portfolio')
    print('{:.2f}% in wvretd and {:.2f}% in b2ret'.format(wrpl[0], wrpl[1]))
    print('Mean of RP-levered portfolio')
    print(mean_wrpl[0])
    print('Standard deviation of RP-levered portfolio')
    print(std_wrpl)
    print('Sharpe ratio of RP-levered portfolio')
    print(sr_wrpl[0])
    
    # plot portfolio, tangency, efficient frontiers
    wvary = np.linspace(0, 1, 101)
    mean_wvary = []
    std_wvary = []
    for w in wvary:
        port = np.ones((1, 2))*[w, 1-w]
        mean_wvary.append(port.dot(mu)[0])
        std_wvary.append(np.sqrt(port.dot(SIGMA.dot(port.T)))[0][0])
    
    fig, ax = plt.subplots()
    plt.plot(std_wvary, mean_wvary)
    plt.scatter(std_wtan, mean_wtan, c = 'r', marker = 'x')
    ax.annotate('Tangency portfolio', xy = (std_wtan, mean_wtan*0.90))
    plt.scatter(std_wport, mean_wport, c = 'r', marker = 'x')
    ax.annotate('60/40 portfolio', xy = (std_wport*1.05, mean_wport*1.05))
    plt.scatter(std_wrpl, mean_wrpl, c = 'r', marker = 'x')
    ax.annotate('RP-levered portfolio', xy = (std_wrpl*0.50, mean_wrpl*1.05))
    plt.scatter(std_wrpu, mean_wrpu, c = 'r', marker = 'x')
    ax.annotate('RP-unlevered portfolio', xy = (std_wrpu*1.1, mean_wrpu*0.98))
    x_plot = np.linspace(0, max(std_wvary), 10000)
    slope = (mean_wtan - R0)/std_wtan
    y_plot = R0 + slope[0]*x_plot
    plt.plot(x_plot, y_plot, 'g')
    
    
# =============================================================================
# Third part
# =============================================================================    
    
    
    # compute rp portfolio using rolling means and standard deviation
    # compute metrics of returns by taking the rolling metrics
    win = 36
    rolling_mean = risky.rolling(window = win, min_periods = win).mean()
    rolling_std = risky.rolling(window = win, min_periods = win).std()
    rolling_cov = risky.rolling(window = win, min_periods = win).cov()
    rolling_excess_std = excess.rolling(window = win, min_periods = win).std()
    rolling_excess_std_inv = rolling_excess_std.apply(lambda x: 1/x)
    rolling_free = free.rolling(window = win, min_periods = win).mean()

    # compute rolling version     
    wrpl_df = pd.DataFrame()
    mean_wrpl_df = pd.DataFrame()
    
    wrpu_df = pd.DataFrame()
    mean_wrpu_df = pd.DataFrame()
    
    for date in risky.index:
        
        # compute weights of RP-unlevered
        k = 1/sum(rolling_excess_std_inv.loc[date])
        temp_wu = k*rolling_excess_std_inv.loc[date]
        temp_wu = np.array([temp_wu.values])        
        temp_wu = pd.DataFrame(data = temp_wu, columns = risky.columns)
        temp_wu.index = [date]
        wrpu_df = wrpu_df.append(temp_wu)
        
        # compute returns on wrpu
        temp_mean = temp_wu.values[0].T.dot(rolling_mean.loc[date].values)
        temp_mean = pd.DataFrame(data = np.array([temp_mean]))
        temp_mean.index = [date]
        mean_wrpu_df = mean_wrpu_df.append(temp_mean)
        
        # compute weights of RP-unlevered
        k = std_wport / std_wrpu
        temp_wl = k*temp_wu    
        temp_wl = temp_wl.values[0]
        temp_wfree = 1 - sum(temp_wl)
        temp_mean = temp_wl.dot(rolling_mean.loc[date].values) + temp_wfree*rolling_free.loc[date].values
        temp_mean = pd.DataFrame(data = np.array([temp_mean]))
        temp_mean.index = [date]
        mean_wrpl_df = mean_wrpl_df.append(temp_mean)

    
    # compute mean
    print('\nMean of returns of RP-unleveverd portfolio using rolling-window')
    print(mean_wrpu_df.mean().values[0])
    
    print('Mean of returns of RP-levered portfolio using rolling-window')
    print(mean_wrpl_df.mean().values[0])


# =============================================================================
# Fourth part
# =============================================================================    
    
    # compute optimal portfolio
    a = 6
    w0 = (SIGMA_inv.dot(mu - R0))/a
    w0free = 1 - sum(w0)
    r0 = w0.dot(mu) + w0free*R0
    sig0 = np.sqrt(w0.dot(SIGMA.dot(w0.T)))
    sr0 = (r0-R0)/sig0
    
    print('\nOptimal portfolio')
    print('{:.2f}% in wvretd and {:.2f}% in b2ret'.format(w0[0], w0[1]))
    print('\nExpected return of optimal portfolio')
    print(r0[0])
    print('Standard deviation of optimal portfolio')
    print(sig0)
    print('Sharpe ratio of optimal portfolio')
    print(sr0[0])
    



    