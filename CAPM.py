from datetime import date
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st
from load_css import local_css
from PIL import Image
import operator
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stocks = ["AAPL", "BA", "MSFT", "TSLA", "T", "MGM", "AMZN","GOOG","^GSPC"]
tickers={"AAPL":"Apple", "BA":"The Boeing Company", "MSFT":"Microsoft", "TSLA":"Tesla", "T":"AT&T Inc", "MGM":"MGM Resorts International", "AMZN":"Amazon","GOOG":"Alphabet Inc.","^GSPC":"S&P 500"}

panda = Image.open("panda.jpg")
formula = Image.open('formula.png')
formula2 = Image.open('formula2.png')
formula3 = Image.open('formula3.png')

beta = {}
alpha = {}
ER = {}
Er={}

#loading CSS
local_css("style.css")

# HTML codes
capm_intro = """
<ul>
    <li>CAPM is one of the most important models in the finance.
    <li>It describes the relationship between systematic risk and expected return for assets, particularly stocks.
    <li>CAPM is widely used throughout finance for pricing risky securities and generating expected returns for assets given the risk of those assets and cost of capital.
</ul>
"""


capm_formula="""
<ul>
    <li>The formula for calculating the expected return of an asset given its risk is as follows:
</ul>
"""


rfrate_content = """
<div><h3>Risk-free rate(Rf)</h3></div>
<ul>
    <li>The risk-free rate of return refers to the theoretical rate of return of an investment with zero risk.
    <li>In practice, the risk-free rate of return does not truly exist, as every investment carries at least a small amount of risk.
    <li>Investors who are extreamely risk averse would prefer to buy the risk free asset to protect their money and earn a low return rf
    <li>If investors are intrested in more return , they have to bear more risk compared to the risk free asset.
</ul>
"""


rmrate_content="""
<div><h3>Market Portfolio(Rm)</h3></div>
<ul>
    <li>A market portfolio is a theoretical, diversified group of every type of investment in the world, with each asset weighted in proportion to its total presence in the market.
    <li>A good representation of the market portfolio is the S&P500. It is one of the most commonly followed equity indices.
    <li>The Standard and Poor's 500, or the S&P 500, is a free-float weighted measurement stock market index of 500 of the largest companies listed on stock exchanges in the United States.
    <li>To know more about S&P 500, <a href = "https://en.wikipedia.org/wiki/S%26P_500"> Click here</a>.
</ul>
"""


beta_content = """
<div><h3>What is Beta ? </h3></div>
<ul>
    <li>Beta represents the slope of the line regression line (market returns vs stock returns).
    <li>It measures the responsiveness of a stock's price to changes in the overall stock market.
    <li> It is used as a measure of risk and is an integral part of the <b>CAPM</b>.
    <li> A company with a higher beta has greater risk and also greater expected returns.
    <li> Properties of Beta:-
    <ul>
        <li> β = 1, exactly as volatile as the market
        <li> β > 1, more volatile than the market
        <li> 0 < β < 1, less volatile than the market
        <li> β = 0, uncorrelated to the market
        <li> β < 0, negatively correlated to the market
    </ul>
</ul>
"""
alpha_content = """
<div><h3>What is alpha ?</h3></div>
<ul>
    <li>In this section, we will try to fit a line between daily returns of the given stock and the market (S&P 500)</li>
    <li>The equation of the linear regression line will be Y = Beta * X + Alpha. Alpha is the Y-intercept of the line.</li>
    <li>It is called <a href = "https://en.wikipedia.org/wiki/Jensen%27s_alpha">Jensen's Alpha</a>. Alpha describes the strategy's ability to beat the market (S&P500)</li>
    <li>Alpha indicates the “excess return” or “abnormal rate of return</li>
    <li>Alpha, often considered the active return on an investment, gauges the performance of an investment against a market index or benchmark that is considered to represent the market’s movement as a whole.</li>
</ul>
"""

footer = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

<div class="footer">
    <p>Developed with ❤️ by Aditya Sinha</p>
    <div class = "social>
        <i class="fa fa-github" aria-hidden="true"></i>
        <a href="https://github.com/aditya113141" class="fa fa-github fa-lg"></a>
        <a href="https://www.linkedin.com/in/adityakumar-sinha-485a40193/" class="fa fa-linkedin fa-lg"></a>
    </div>
    <p> Copyright © 2021 Adityakumar Sinha </p>
</div>
"""

#Utility Functions

#load stock data from Yahoo Finance
def load_data(stocks):
    datas = []
    for ticker in stocks:
        data = yf.download(ticker,START, TODAY)
        data.reset_index(inplace=True)
        datas.append(data)
    df = pd.DataFrame()
    df['Date'] = datas[0]['Date']
    for i in range(len(datas)):
        df[stocks[i]] = datas[i]["Adj Close"]
    return df

#Plotting the raw data
def plot_raw_data(data):
    fig = go.Figure()
    for stock in data.columns[1:]:
        fig.add_trace(go.Scatter(x=data['Date'], y=data[stock], name=tickers[stock]))
    fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Calculating the daily return
def daily_return(df):
  df_daily_return = df.copy()
  for i in df.columns[1:]:
    for j in range(1, len(df)):
      df_daily_return[i][j] = ((df[i][j]- df[i][j-1])/df[i][j-1]) * 100
    df_daily_return[i][0] = 0
  return df_daily_return

def fun(beta):
    if beta > 1:
        return "more volatile than the market"
    elif beta == 1:
        return "exactly as volatile as the market"
    elif beta > 0:
        return "less volatile than the market"
    elif beta == 0:
        return "uncorrelated to the market"
    else:
        return "negatively correlated to the market"


def Fun1(returns,rev):
    temp = []
    for it in returns:
        temp.append((it,returns[it]))
    temp.sort(key = lambda x:x[1],reverse=rev)
    ret = []
    for i in range(3):
        ret.append(temp[i][0])
    return ret

# doing CAPM Analysis of Stocks
def cal_capm():
    st.subheader("Calculating annualized rate of return for every stock using CAPM")
    st.latex(r'ER_i = R_f + \beta_i( ER_m -R_f) ' )
    for ticker in tickers:
        if ticker != "^GSPC":
            er = round(rf + beta[tickers[ticker]]*(rm-rf),3)
            Er[tickers[ticker]] = er
            ER[tickers[ticker]] = str(er) + "%"
    st.write(ER)

def return_check(x):
    if x >= rm:
        return "Suggested to invest, returns better than S&P 500"
    else:
        return "Not suggested to invest, doesn't returns better than S&P 500"


 # The main code starts here

st.sidebar.image(panda)

#General Introduction
st.title("Capital Asset Pricing Model")
st.markdown(capm_intro,unsafe_allow_html=True)
st.header(" Understanding the Capital Asset Pricing Model (CAPM)")
st.markdown(capm_formula,unsafe_allow_html=True)
st.latex(r'ER_i = R_f + \beta_i( ER_m -R_f) ' )
st.image(formula)
st.markdown(rfrate_content,unsafe_allow_html=True)
st.markdown(rmrate_content,unsafe_allow_html=True)
st.markdown(beta_content,unsafe_allow_html=True)

#Loading live Stock Data
st.header("Load live Stock Data")
curr_stocks = load_data(stocks)
st.write(curr_stocks.tail(15))
st.subheader("Ticker Key")

# Adding ticker Key
ticker_key = "<ul>"
for ticker in tickers:
    ticker_key = ticker_key + f'<li><h4>{ticker} : {tickers[ticker]}</h4></li>'
ticker_key =  ticker_key + "</ul>"
st.write(tickers)

#Plotting Raw Data
plot_raw_data(curr_stocks)

#Calculating Daily Returns
st.header("Calculating Daily Returns")
st.image(formula2)
stocks_daily_return = daily_return(curr_stocks)
st.write(stocks_daily_return.tail(15))

#Calculating Average Daily Return
st.write("Let's check the average daily returns ")
means = stocks_daily_return.mean()
st.write(means)
means_dummy = means.copy()
means_dummy.drop(columns=["^GSPC"])
st.write(f"The average return of S&P 500 is {round(means['^GSPC'],3)}.")
st.write(f"Amoung other stocks, {tickers[max(means_dummy.items(), key=operator.itemgetter(1))[0]]} has the highest average return and {tickers[min(means_dummy.items(), key=operator.itemgetter(1))[0]]} has the lowest average return.")



# Calcuating Alpha and Beta
st.header("Calculating Alpha and Beta for each stock")
st.markdown(alpha_content,unsafe_allow_html=True)
for ticker in tickers:
    if ticker != "^GSPC":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stocks_daily_return["^GSPC"], y=stocks_daily_return[ticker], name=tickers[ticker],mode='markers'))
        b, a = np.polyfit(stocks_daily_return["^GSPC"], stocks_daily_return[ticker], 1)
        beta[tickers[ticker]]=b
        alpha[tickers[ticker]]=a
        fig.add_trace(go.Line(x=stocks_daily_return["^GSPC"], y = b*stocks_daily_return["^GSPC"] + a))
        fig.layout.update(title_text = tickers[ticker], xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)
        st.write(f"Beta = {round(b,3)}, Alpha = {round(a,3)} ")
        st.write(f"{tickers[ticker]} stocks are {fun(b)} ")
        
#Calculating Average Expected return for each stock
st.header("Calculating expected return for each stock")
rf = 1.63
rm = round(means['^GSPC']*251,3)
st.write(f"The current risk free rate (Rf) is {rf}%")
st.subheader("Let's calculate the annualized rate of return for S&P500 ")
st.write("Note that out of 365 days/year, stock exchanges are closed for 104 days during weekend days (Saturday and Sunday) ")
st.image(formula3)
st.write(f"Thus, the current expected value of Rm is {rm}%")
cal_capm()

#The content of the subheader tells everything,LOL
st.subheader("Calculating returns for different portfolio combinations")
port = "<ul>"
equal_port = f"""
    <ul>
    <li><u>Assuming a portfolio with all stocks in equal weight</u><br> 
        Expected return = ( 1 / {len(Er)} ) * Sum of expected returns of individual stocks = <b>{round(sum(Er.values())/len(Er),3)}%</b><br>
        <b>{return_check(sum(Er.values())/len(Er))}</b>
    </li>
    </ul>
"""
consumer_port = f"""
    <ul>
    <li><u>Assuming a portfolio of Consumer Services ( MGM and AT&T) having equal weights</u><br>
        Expected return = ( 1 / 2 ) * Sum of expected returns of MGM and AT&T stocks = <b>{round((Er[tickers["MGM"]] + Er[tickers["T"]])/2,3)}%</b><br>
        <b>{return_check((Er[tickers["MGM"]] + Er[tickers["T"]])/2)}</b>
    </li>
    </ul>
"""
manu_port = f"""
    <ul>
    <li><u>Assuming a portfolio of Manufacturing Sector( Tesla and Boeing Company) having equal weights</u><br>
        Expected return = ( 1 / 2 ) * Sum of expected returns of Tesla and Boeing Company stocks = <b>{round((Er[tickers["TSLA"]] + Er[tickers["BA"]])/2,3)}%</b><br>
        <b>{return_check((Er[tickers["TSLA"]] + Er[tickers["BA"]])/2)}</b>
    </li>
    </ul>
"""
personal_port = f"""
    <ul>
    <li><u>Assuming a portfolio of personal devices sector(Alphabet, Apple, Microsoft, Amazon) having equal weights</u><br>
        Expected return = ( 1 / 4 ) * Sum of expected returns of Alphabet, Apple, Microsoft and Amazon stocks = <b>{round((Er[tickers["AAPL"]] + Er[tickers["AMZN"]] + Er[tickers["GOOG"]] + Er[tickers["MSFT"]])/4,3)}%</b><br>
        <b>{return_check((Er[tickers["AAPL"]] + Er[tickers["AMZN"]] + Er[tickers["GOOG"]] + Er[tickers["MSFT"]])/4)}</b>
    </li>
    </ul>
"""

high = Fun1(Er,True)
low  = Fun1(Er,False)

high_port = f"""
    <ul>
    <li><u>Assuming a portfolio of top 3 high performing stocks({high[0]}, {high[1]} and {high[2]}) having equal weights</u><br>
        Expected return = ( 1 / 3 ) * Sum of expected returns of {high[0]}, {high[1]} and {high[2]} stocks = <b>{round((Er[high[0]] + Er[high[1]] + Er[high[2]])/3,3)}%</b><br>
        <b>{return_check((Er[high[0]] + Er[high[1]] + Er[high[2]])/3)}</b>
    </li>
    </ul>
"""
low_port = f"""
    <ul>
    <li><u>Assuming a portfolio of top 3 high performing stocks({low[0]}, {low[1]} and {low[2]}) having equal weights</u><br>
        Expected return = ( 1 / 3 ) * Sum of expected returns of {low[0]}, {low[1]} and {low[2]} stocks = <b>{round((Er[low[0]] + Er[low[1]] + Er[low[2]])/3,3)}%</b><br>
        <b>{return_check((Er[low[0]] + Er[low[1]] + Er[low[2]])/3)}</b>
    </li>
    </ul>
"""

port = equal_port + personal_port + manu_port + consumer_port + high_port + low_port
st.markdown(port,unsafe_allow_html=True)

#Again, LOL
st.header("Check the performance of your own choice of portfolio")

selected_stocks = st.multiselect("Select your stocks", list(Er.keys()))
button = st.button("Check the performance")

if button and len(selected_stocks) > 0:
    summation = 0
    for stock in selected_stocks:
        summation = summation + Er[stock]
    expected_return = summation/len(selected_stocks)
    st.write(f"Expected return is {round(expected_return,3)}")
    st.write(return_check(expected_return))


st.markdown(footer,unsafe_allow_html=True)
