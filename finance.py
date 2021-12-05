import pandas as pd
import numpy as np
from datetime import timedelta
import yfinance as yf
import quandl
from scipy.optimize import minimize
import plotly.graph_objs as go

# The code is mainly taken from the Coursera class https://www.coursera.org/specializations/investment-management-python-machine-learning
# I added and customized the code to my belongings

DAX = {"Adidas": "ADS.DE", "Airbus": "AIR.DE", "Allianz": "ALV.DE", "BASF": "BAS.DE", "Bayer": "BAYN.DE",
       "Beiersdorf": "BEI.DE", "BMW": "BMW.DE", "Brenntag": "BNR.DE", "Continental": "CON.DE", "Covestro": "1COV.DE",
       "Daimler": "DAI.DE", "Delivery Hero": "DHER.DE", "Deutsche Bank": "DBK.DE", "Deutsche Börse": "DB1.DE",
       "Deutsche Post": "DPW.DE", "Deutsche Telekom": "DTE.DE", "E.ON": "EOAN.DE", "Fresenius": "FRE.DE",
       "Fresenius Medical Care": "FME.DE", "HeidelbergCelement": "HEI.DE", "HelloFresh": "HFG.DE", "Henkel": "HEN3.DE",
       "Infineon": "IFX.DE", "Linde": "LIN.DE", "Merck": "MRK.DE", "MTU": "MTX.DE", "Munich Re": "MUV2.DE",
       "Porsche SE": "PAH3.DE", "Puma": "PUM.DE", "Qiagen": "QIA.DE", "RWE": "RWE.DE", "SAP": "SAP.DE",
       "Sartorius": "SRT3.DE", "Siemens": "SIE.DE", "Siemens Energy": "ENR.DE", "Siemens Healthineers": "SHL.DE",
       "Symrise": "SY1.DE", "Volkswagen": "VOW3.DE", "Vonovia": "VNA.DE", "Zalando": "ZAL.DE"}

NASDAQ100 = {"Activision Blizzard": "ATVI", "Adobe": "ADBE", "Advanced Micro Devices": "AMD",
             "Align Technology": "ALGN",
             "Alphabet (Class A)": "GOOGL", "Alphabet (Class C)": "GOOG", "Amazon.com": "AMZN",
             "American Electric Power": "AEP",
             "Amgen": "AMGN", "Analog Devices": "ADI", "Ansys": "ANSS", "Apple": "AAPL", "Applied Materials": "AMAT",
             "ASML Holding": "ASML", "Atlassian": "TEAM", "Autodesk": "ADSK", "Automatic Data Processing": "ADP",
             "Baidu": "BIDU",
             "Biogen": "BIIB", "Booking Holdings": "BKNG", "Broadcom": "AVGO", "Cadence Design Systems": "CDNS",
             "CDW": "CDW",
             "Cerner": "CERN", "Charter Communications": "CHTR", "Check Point": "CHKP", "Cintas": "CTAS",
             "Cisco Systems": "CSCO",
             "Cognizant": "CTSH", "Comcast": "CMCSA", "Copart": "CPRT", "Costco": "COST", "CrowdStrike": "CRWD",
             "CSX Corporation": "CSX",
             "DexCom": "DXCM", "DocuSign": "DOCU", "Dollar Tree": "DLTR", "eBay": "EBAY", "Electronic Arts": "EA",
             "Exelon": "EXC",
             "Fastenal": "FAST", "Fiserv": "FISV", "Fox Corporation (Class A)": "FOXA",
             "Fox Corporation (Class B)": "FOX",
             "Gilead Sciences": "GILD", "Honeywell": "HON", "Idexx Laboratories": "IDXX", "Illumina": "ILMN",
             "Incyte": "INCY",
             "Intel": "INTC", "Intuit": "INTU", "Intuitive Surgical": "ISRG", "JD.com": "JD", "Keurig Dr Pepper": "KDP",
             "KLA Corporation": "KLAC", "Kraft Heinz": "KHC", "Lam Research": "LRCX", "Lululemon Athletica": "LULU",
             "Marriott International": "MAR", "Marvell Technology": "MRVL", "Match Group": "MTCH",
             "MercadoLibre": "MELI",
             "Meta Platforms": "FB", "Microchip Technology": "MCHP", "Micron Technology": "MU", "Microsoft": "MSFT",
             "Moderna": "MRNA",
             "Mondelēz International": "MDLZ", "Monster Beverage": "MNST", "NetEase": "NTES", "Netflix": "NFLX",
             "Nvidia": "NVDA",
             "NXP Semiconductors": "NXPI", "O'Reilly Automotive": "ORLY", "Okta": "OKTA", "Paccar": "PCAR",
             "Paychex": "PAYX",
             "PayPal": "PYPL", "Peloton Interactive": "PTON", "PepsiCo": "PEP", "Pinduoduo": "PDD", "Qualcomm": "QCOM",
             "Regeneron Pharmaceuticals": "REGN", "Ross Stores": "ROST", "Seagen": "SGEN", "Sirius XM": "SIRI",
             "Skyworks Solutions": "SWKS", "Splunk": "SPLK", "Starbucks": "SBUX", "Synopsys": "SNPS",
             "T-Mobile US": "TMUS",
             "Tesla": "TSLA", "Texas Instruments": "TXN", "Trip.com Group": "TCOM", "Verisign": "VRSN",
             "Verisk Analytics": "VRSK",
             "Vertex Pharmaceuticals": "VRTX", "Walgreens Boots Alliance": "WBA", "Workday": "WDAY",
             "Xcel Energy": "XEL",
             "Xilinx": "XLNX", "Zoom Video Communications": "ZM"}


def get_riskfree_rate(start_date='2010-01-01'):
    """
    :param start_date: first date the risk free rate is pulled
    :return: returns risk free rate per day, gets data from quandl (26 week T Bills)

    Data comes daily with annual rate and is therefor transformed to a daily basis
    """

    # get risk free rate from quandl
    risk_free = quandl.get("USTREASURY/BILLRATES", start_date=start_date)
    # calculate on daily basis
    risk_free['risk_free_rate'] = (1 + risk_free['26 Wk Bank Discount Rate'] / 100) ** (1 / 360) - 1
    return risk_free['risk_free_rate']


def get_returns(ticker, start_date='2010-01-01'):
    """
    :param ticker: list of all tickers who's returns should be pulled
    :param start_date: first date
    :return: returns dataframe with daily return of all tickers
    """

    # create final df
    df_prices = pd.DataFrame()

    # get price data for every stock
    for stock in ticker:
        tick = yf.Ticker(stock)

        # get historical market data
        hist = tick.history(start=start_date)
        df_prices = df_prices.join(hist['Close'], how='outer', rsuffix=stock)

    df_prices.columns = ticker

    # in the case a price is missing for one stock, fill with the price above
    df_prices[df_prices.loc[:, :] == ""] = np.nan
    df_prices = df_prices.fillna(method='ffill')

    return df_prices


def compound(r):
    """
    :param r: DataSeries with return data
    :return: returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())


def annualize_rets(r, periods_per_year):
    """
    :param r: DataSeries with return data
    :param periods_per_year: should be 260 for daily, 12 for monthly and 4 for quarterly data
    :return: Annualizes a set of returns
    """
    compounded_growth = 1 + compound(r)
    n_periods = r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1


def annualize_vol(r, periods_per_year):
    """
    :param r: DataSeries with return data
    :param periods_per_year: should be 260 for daily, 12 for monthly and 4 for quarterly data
    :return: Annualizes the vol of a set of returns
    """
    return r.std() * (periods_per_year ** 0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    :param r: DataSeries with return data
    :param riskfree_rate: DataSeries with risk free rate data, same periods as r
    :param periods_per_year: should be 260 for daily, 12 for monthly and 4 for quarterly data
    :return: Computes the annualized sharpe ratio of a set of returns
    """

    excess_ret = r - riskfree_rate
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol


def portfolio_return(weights, returns):
    """
    :param weights: weights are a numpy array or Nx1 matrix
    :param returns: returns are a numpy array or Nx1 matrix
    :return: Computes the return on a portfolio from constituent returns and weights
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    :param weights:  weights are a numpy array or N x 1 maxtrix
    :param covmat: covmat is an N x N matrix
    :return: Computes the vol of a portfolio from a covariance matrix and constituent weights
    """
    return (weights.T @ covmat @ weights) ** 0.5


def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    # number of stocks
    n = er.shape[0]
    # all with the same weight
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
                        }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
                        }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1, return_is_target),
                       bounds=bounds)
    return weights.x


def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    # number of stocks
    n = er.shape[0]
    # all with the same weight
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
                        }

    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol

    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x


def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def optimal_weights(n_points, er, cov):
    """
    Returns a list of weights that represent a grid of n_points on the efficient frontier
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def plot_ef(df, layout, riskfree_rate=0):
    """
    :param df: price data for stocks
    :param layout: layout for chart
    :param riskfree_rate: yearly return of a riskfree asset
    :return: Plots the multi-asset efficient frontier as a plotly go figure, adds different types of optimal
    portfolios as stars
    """

    # create expected returns and cov matrix
    er = annualize_rets(df.pct_change(), 260)
    cov = df.pct_change().dropna().cov()

    # create a return/ volatility matrix for 20 weights
    weights = optimal_weights(20, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    df_ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })

    # MSR-Portfolio
    w_msr = msr(riskfree_rate, er, cov)
    r_msr = portfolio_return(w_msr, er)
    vol_msr = portfolio_vol(w_msr, cov)

    # EW-Portfolio
    n = er.shape[0]
    w_ew = np.repeat(1 / n, n)
    r_ew = portfolio_return(w_ew, er)
    vol_ew = portfolio_vol(w_ew, cov)

    # GMV-Portfolio
    w_gmv = gmv(cov)
    r_gmv = portfolio_return(w_gmv, er)
    vol_gmv = portfolio_vol(w_gmv, cov)

    # ERC-Portfolio
    w_erc = weight_erc(df.pct_change().dropna())
    r_erc = portfolio_return(w_erc, er)
    vol_erc = portfolio_vol(w_erc, cov)

    data = []
    # add price line
    trace = go.Scatter(x=df_ef['Volatility'], y=df_ef['Returns'], line=dict(color='#fec036'), name='Effective Frontier')
    data.append(trace)
    trace_2 = go.Scatter(x=[vol_gmv], y=[r_gmv], mode='markers',
                         marker=dict(
                             color='rgb(217,72,1)',
                             size=20,
                             symbol='star',
                             line=dict(
                                 color='black',
                                 width=2
                             )
                         ),
                         name='Global Minimum Vol')
    data.append(trace_2)

    # add msr portfolio
    trace_3 = go.Scatter(x=[vol_msr], y=[r_msr], mode='markers',
                         marker=dict(
                             color='rgb(151, 103, 58)',
                             size=20,
                             symbol='star',
                             line=dict(
                                 color='black',
                                 width=2
                             )
                         ),
                         name='Maximum Sharpe Ratio')
    data.append(trace_3)

    # add equal weight portfolio
    trace_4 = go.Scatter(x=[vol_ew], y=[r_ew], mode='markers',
                         marker=dict(
                             color='rgb(254,196,79)',
                             size=20,
                             symbol='star',
                             line=dict(
                                 color='black',
                                 width=2
                             )
                         ),
                         name='Equally Weighted')
    data.append(trace_4)

    # add erc portfolio
    trace_5 = go.Scatter(x=[vol_erc], y=[r_erc], mode='markers',
                         marker=dict(
                             color='rgb(57, 45, 37)',
                             size=20,
                             symbol='star',
                             line=dict(
                                 color='black',
                                 width=2
                             )
                         ),
                         name='Equally Risk Contribution')
    data.append(trace_5)

    # create figure
    fig = go.Figure(data=data, layout=layout)

    return fig


def sample_cov(r, **kwargs):
    """
    Returns the sample covariance of the supplied returns
    """
    return r.cov()


def risk_contribution(w, cov):
    """
    Compute the contributions to risk of the constituents of a portfolio, given a set of portfolio weights and a covariance matrix
    """
    total_portfolio_var = portfolio_vol(w, cov) ** 2
    # Marginal contribution of each constituent
    marginal_contrib = cov @ w
    risk_contrib = np.multiply(marginal_contrib, w.T) / total_portfolio_var
    return risk_contrib


def target_risk_contributions(target_risk, cov):
    """
    Returns the weights of the portfolio that gives you the weights such
    that the contributions to portfolio risk are as close as possible to
    the target_risk, given the covariance matrix
    """
    n = cov.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
                        }

    def msd_risk(weights, target_risk, cov):
        """
        Returns the Mean Squared Difference in risk contributions
        between weights and target_risk
        """
        w_contribs = risk_contribution(weights, cov)
        return ((w_contribs - target_risk) ** 2).sum()

    weights = minimize(msd_risk, init_guess,
                       args=(target_risk, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x


def equal_risk_contributions(cov):
    """
    Returns the weights of the portfolio that equalizes the contributions
    of the constituents based on the given covariance matrix
    """
    n = cov.shape[0]
    return target_risk_contributions(target_risk=np.repeat(1 / n, n), cov=cov)


def weight_erc(r, cov_estimator=sample_cov, **kwargs):
    """
    Produces the weights of the ERC portfolio given a covariance matrix of the returns
    """
    est_cov = cov_estimator(r, **kwargs)
    return equal_risk_contributions(est_cov)


def weight_gmv(r, cov_estimator=sample_cov, **kwargs):
    """
    Produces the weights of the GMV portfolio given a covariance matrix of the returns
    """
    est_cov = cov_estimator(r, **kwargs)
    return gmv(est_cov)


def weight_msr(r, cov_estimator=sample_cov, **kwargs):
    """
    Produces the weights of the maximum sharpe ratio portfolio given a covariance matrix of the returns
    """
    est_cov = cov_estimator(r, **kwargs)
    ann_rets = annualize_rets(r, 260)
    return msr(0, ann_rets, est_cov)


def get_weights_erc_gmv(r, rebalance_period):
    """
    :param r: dataframe of prices of stocks
    :param rebalance_period: How often do the user rebalance the portfolio. Possible values:
            [daily, weekly, monthly, quarterly, yearly, once]
    :return: returns two dataframes, first with the weights of the erc-weighting for each stock, the second with the
            weights of gmv-weighting
    """

    # convert into returns
    r = r.pct_change()
    r = r.iloc[1: , :]

    # check for period --> save dates in which weights should be calculated (rebalancing dates)
    if rebalance_period == "daily":
        dates = r.index
    elif rebalance_period == "weekly":
        dates = r.groupby([r.index.year, r.index.week]).head(1).index
    elif rebalance_period == "monthly":
        dates = r.groupby([r.index.year, r.index.month]).head(1).index
    elif rebalance_period == "quarterly":
        dates = r.groupby([r.index.year, r.index.quarter]).head(1).index
    elif rebalance_period == "yearly":
        dates = r.groupby([r.index.year]).head(1).index
    elif rebalance_period == "once":
        dates = r.loc[0, :].index
    else:
        dates = []

    erc = []
    gmv_list = []
    # run weighting for each rebalancing date
    for index, row in r.iterrows():
        if index in dates:
            # lookback period for variance is 2 years
            df = r.loc[index - timedelta(days=365 * 2):index, :]

            # if stock was not on the market yet, drop it ad pass weight 0
            missing_price = df.columns[df.iloc[-1:, :].isna().any()].tolist()
            missing_col_nr = [df.columns.get_loc(c) for c in missing_price]
            df = df.drop(missing_price, axis=1)
            # append weights for stocks in erc and gmv portfolios
            erc_weights = weight_erc(df).tolist()
            gmv_weights = weight_gmv(df).tolist()
            for stock in missing_col_nr:
                erc_weights.insert(stock, 0)
                gmv_weights.insert(stock, 0)
            erc.append([i for i in erc_weights])
            gmv_list.append([i for i in gmv_weights])
    # create final dataframes
    erc_df = pd.DataFrame(erc, columns=[col for col in r], index=dates)
    gmv_df = pd.DataFrame(gmv_list, columns=[col for col in r], index=dates)
    return erc_df, gmv_df
