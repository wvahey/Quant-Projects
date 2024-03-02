import os
import sys
import numpy as np
import pandas as pd
import scipy.optimize as sc
import plotly.graph_objects as go
import yfinance as yfin
from pandas_datareader import data as pdr
from statistics import stdev

# Line required for pdr.get_data_yahoo() to function properly
yfin.pdr_override()

# Get the tickers of current S&P500 stock list on Wikipedia
stockInfo = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers_np = stockInfo['Symbol'].to_numpy()

# Exclude stocks that do not have full data during analysis period
excludeStocks = np.array(['VLTO', 'BF.B', 'BRK.B', 'ABNB', 'CARR', 'KVUE', 'GEHC', 'OTIS', 'CEG', 'RHI', 'CTLT'])
stockList = [stock for stock in tickers_np[:50] if stock not in excludeStocks] #limiting stocks to 50

# Number of trading days durnig a fiscal year
NB_TRADING_DAYS_PER_YEAR = 252

# Indicate the minimum & maximum weight for each stock in portfolio: between 0 and 10%
constraintSet = (0, 0.1)

# Start & end dates of the analysed period
startDate = '2019-01-01'
endDate = '2021-12-31'

# Get stocks' adjusted closing prices & calculate covariance matrix
# The adjusted closing price includes anything that would affect the stock price (stock splits, dividends...)
def getData(stocks: list, start: str, end: str):
    
    stockPricesDf = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockPricesDf = stockPricesDf['Adj Close']

    # Exclude columns that have at least one missing stock price
    stockPricesDf = stockPricesDf.dropna(axis='columns', how='any')

    dailyStockPriceChanges = stockPricesDf.pct_change(fill_method=None)
    covMatrixDf = dailyStockPriceChanges.cov()

    return stockPricesDf, covMatrixDf

# Calculate annualised stock return over a specific period
def stockAnnualisedReturn(stockPricesDf: pd.Series):
        
    initialPrice = stockPricesDf.iloc[0]
    finalPrice = stockPricesDf.iloc[-1]
    totalReturn = (finalPrice - initialPrice) / initialPrice
    annualisedStockReturn = totalReturn * NB_TRADING_DAYS_PER_YEAR / len(stockPricesDf)
    annualisedStockReturn = round(annualisedStockReturn*100,2)
    return annualisedStockReturn

# Calculate annualised stock variance over a specific period
def stockAnnualisedVariance(stockPricesDf: pd.Series):

    # Daily percentage changes in stock price
    dailyStockPriceChanges = stockPricesDf.pct_change(fill_method=None)

    # Daily stock price volatility
    dailyVolatility = stdev(dailyStockPriceChanges[1:])
    annualisedDailyVolatilityInTradingDays = dailyVolatility * np.sqrt(NB_TRADING_DAYS_PER_YEAR)
    
    annualisedDailyVolatilityInTradingDays = round(annualisedDailyVolatilityInTradingDays*100,2)
    return annualisedDailyVolatilityInTradingDays

# Calculate portfolio annualised return over a specific period
def portfolioAnnualisedReturn(weights: np.ndarray, stockPricesDf: pd.DataFrame):
    
    annualisedPortfolioReturn = 0
    annualisedReturnsPerStock_array = []
    for stock in stockPricesDf:

        # For each stock, calculate annualised return over all period of stockPricesDf
        initialPrice = stockPricesDf[stock].iloc[0]
        finalPrice = stockPricesDf[stock].iloc[-1]
        totalReturn = (finalPrice - initialPrice) / initialPrice
        annualisedReturnsPerStock_array.append(totalReturn * NB_TRADING_DAYS_PER_YEAR / len(stockPricesDf))

    # Calculate annualised portfolio return
    for w, r in zip(weights, annualisedReturnsPerStock_array): annualisedPortfolioReturn += w * r
    return annualisedPortfolioReturn

# Calculate portfolio annualised variance over a specific period
def portfolioAnnualisedVariance(weights: np.ndarray, covMatrixDf: pd.DataFrame):

    # Calculate portfolio annualised daily volatility (in trading days)
    pAannualisedDailyVolatilityInTradingDays = 0
    pAannualisedDailyVolatilityInTradingDays = np.sqrt(np.dot(weights.T, np.dot(covMatrixDf, weights)))*np.sqrt(NB_TRADING_DAYS_PER_YEAR)
    
    return pAannualisedDailyVolatilityInTradingDays

# For each returnTarget, Minimise variance by altering the weights of the portfolio
def efficientOpt(stockPricesDf: pd.DataFrame, covMatrixDf: pd.DataFrame, returnTarget: float):

    numAssets = len(stockPricesDf.columns)
    args = (covMatrixDf)
    constraints = ({'type':'eq', 'fun': lambda x: portfolioAnnualisedReturn(x, stockPricesDf) - returnTarget},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraintSet for asset in range(numAssets))
    optimalPortfolio = sc.minimize(portfolioAnnualisedVariance, numAssets*[1./numAssets], 
                                   args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)

    return optimalPortfolio

# Calculate negative Sharpe ratio of the overall portfolio
def negativeSR(weights: np.ndarray, stockPricesDf: pd.DataFrame, covMatrixDf: pd.DataFrame):

    pAnnualisedReturn = portfolioAnnualisedReturn(weights, stockPricesDf)
    pAannualisedDailyVolatilityInTradingDays = portfolioAnnualisedVariance(weights, covMatrixDf)
    return - pAnnualisedReturn / pAannualisedDailyVolatilityInTradingDays

# Maximise Sharpe ratio by altering the weights of the portfolio
def maximiseSR(stockPricesDf: pd.DataFrame, covMatrixDf: pd.DataFrame):

    numAssets = len(stockPricesDf.columns)
    args = (stockPricesDf, covMatrixDf)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraintSet for asset in range(numAssets))

    # initial guess of asset optimal weights is equal distribution of assets
    maxSRportfolio = sc.minimize(negativeSR, numAssets*[1./numAssets],
                                 args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return maxSRportfolio

# Minimise variance by altering the weights of the portfolio
def minimizeVariance(stockPricesDf: pd.DataFrame, covMatrixDf: pd.DataFrame):

    numAssets = len(stockPricesDf.columns)
    args = (covMatrixDf)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraintSet for asset in range(numAssets))

    # initial guess of asset optimal weights is equal distribution of assets
    minVolPortfolio = sc.minimize(portfolioAnnualisedVariance, numAssets*[1./numAssets],
                                  args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return minVolPortfolio

# Function to remove elements from the start and end of the array if they have the same value
def removeDuplicateEnds(x_list, y_list):
    start, end = 0, len(x_list) - 1

    # Remove duplicates from the start
    while start < end and round(x_list[start], 4) == round(x_list[start + 1], 4):
        start += 1

    # Remove duplicates from the end
    while end > start and round(x_list[end], 4) == round(x_list[end - 1], 4):
        end -= 1

    # Slicing the arrays to exclude the duplicate elements
    return x_list[start:end + 1], y_list[start:end + 1]

# Return a graph ploting the min volatility, max SR and efficient frontier
def efficientFrontierGraph(
        maxSR_return: float, maxSR_std: float, 
        minVol_return: float, minVol_std: float, 
        volatilityPerTargetReturn: list, targetReturns: np.ndarray,
        stocksReturns: list, stocksStd: list
    ):

    # Maximum Sharpe ratio portfolio
    MaxSharpeRatio = go.Scatter(
        name='Maximium Sharpe ratio',
        mode='markers',
        x=[maxSR_std],
        y=[maxSR_return],
        marker=dict(color='red',size=14,line=dict(width=3, color='black'))
    )

    # Minimum volatility portfolio
    MinVol = go.Scatter(
        name='Mininium Volatility',
        mode='markers',
        x=[minVol_std],
        y=[minVol_return],
        marker=dict(color='green',size=14,line=dict(width=3, color='black'))
    )

    # Efficient Frontier
    EF_curve = go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=[round(ef_std*100, 2) for ef_std in volatilityPerTargetReturn],
        y=[round(target*100, 2) for target in targetReturns],
        line=dict(color='black', width=3, dash='dot')
    )

    data = [EF_curve, MaxSharpeRatio, MinVol]
    layout = go.Layout(
        title = 'Portfolio Optimisation - Efficient Frontier, '+str(len(stockList))+' stocks, ('+startDate+' to '+endDate+')',
        yaxis = dict(title='Annualised Return (%)'),
        xaxis = dict(title='Annualised Daily Volatility (in trading days) (%)'),
        showlegend = True,
        legend = dict(
            x = 0.75, y = 0, traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2),
        width=1400,
        height=600)

    # One marker per stock
    for i in range(len(stockList)):
        StockMarker = go.Scatter(
            name=stockList[i],
            mode='markers',
            x=[stocksStd[i]],
            y=[stocksReturns[i]],
            marker=dict(color='#d1c7c7',size=9,line=dict(width=1, color='#d1c7c7'))
        )
        data.append(StockMarker)

    fig = go.Figure(data=data, layout=layout)
    return fig.show()


# Get stock prices & calculate covariance matrix
stockPrices, covMatrix = getData(stockList, start=startDate, end=endDate)
stockList = stockPrices.columns

# Calculate return & volatility of each stock
stocksReturns = list()
stocksStd = list()
for stock in stockList:

    annualisedReturn = stockAnnualisedReturn(stockPrices[stock])
    annualisedDailyVolatilityInTradingDays = stockAnnualisedVariance(stockPrices[stock])
    stocksReturns.append(annualisedReturn)
    stocksStd.append(annualisedDailyVolatilityInTradingDays)

# Compute max Sharpe ratio portfolio by calling function built earlier
maxSRportfolio = maximiseSR(stockPrices, covMatrix)

# Interrupt the program if the optimisation has failed
if not maxSRportfolio.success: sys.exit('Program interrupted! Failed to compute the maximum Sharpe ratio portfolio!')

# Weights of max Sharpe ratio portfolio
maxSR_weights = np.array(maxSRportfolio.x)

# Composition of max Sharpe ratio portfolio
maxSRportfolio_df = pd.DataFrame(data=[maxSRportfolio.x], columns=covMatrix.columns).T
maxSRportfolio_df = maxSRportfolio_df.sort_values(0, ascending=False)

print('\nOptimal portfolio composition to maximise the Sharpe ratio:\n')
for stock, weight in maxSRportfolio_df.iterrows(): print('\t', stock, ':', round(100*weight[0],2), '%')

# Calculate return & volatility of max Sharpe ratio portfolio using functions built earlier
maxSR_return = portfolioAnnualisedReturn(maxSR_weights, stockPrices)
maxSR_std = portfolioAnnualisedVariance(maxSR_weights, covMatrix)
maxSR_return, maxSR_std = round(maxSR_return*100,2), round(maxSR_std*100,2)

# Compute min volatility portfolio by calling function built earlier
minVolPortfolio = minimizeVariance(stockPrices, covMatrix)

# Interrupt the program if the optimisation has failed
if not minVolPortfolio.success: sys.exit('Program interrupted! Failed to compute the minimum volatility portfolio!')

# Weights of min volatility portfolio
minVolPortfolio_weights = np.array(minVolPortfolio.x)

# Calculate return & volatility of min volatility portfolio using functions built earlier
minVol_return = portfolioAnnualisedReturn(minVolPortfolio_weights, stockPrices)
minVol_std = portfolioAnnualisedVariance(minVolPortfolio_weights, covMatrix)
minVol_return, minVol_std = round(minVol_return*100,2), round(minVol_std*100,2)

# Define list of target returns for efficient frontier
frontierMinReturn = 0.5 * min(0, minVol_return - maxSR_return) / 100
frontierMaxReturn = 2 * maxSR_return / 100
targetReturns = np.linspace(frontierMinReturn, frontierMaxReturn, 20)

# Calculate minimum volatility for each target return
volatilityPerTargetReturn = []
for returnTarget in targetReturns:
    volatilityPerTargetReturn.append(efficientOpt(stockPrices, covMatrix, returnTarget)['fun'])

# Trim the efficient frontier by removing the vertical parts
volatilityPerTargetReturn, targetReturns = removeDuplicateEnds(volatilityPerTargetReturn, targetReturns)

# Plot efficient frontier
efficientFrontierGraph(
    maxSR_return, maxSR_std, 
    minVol_return, minVol_std, 
    volatilityPerTargetReturn, targetReturns,
    stocksReturns, stocksStd
)