# Dash Portfolio Performance Analyzer

## About the app

A Dash application that helps construction your portfolio. 

### Step 1

Choose your parameters on the left side. First, choose the rebalancing period, meaning how often the portfolio should be rebalanced. 
Next, select all the stocks you want to put into your portfolio. For now, stocks from the American Nasdaq 100 and the German DAX40
can be picked. Last, put in a start date. The calculation always starts with pressing the button on the bottum. 

### Step 2
Analyze your portfolio. You have the option to look into several weighting types, for example equally weighted or the global minimum 
variance portfolio. Choosing so, the app shows you yearly returns, variance, the sharpe ratio and the maximal drawdown in the 
selected period. 

### Step 3
Looking into the charts. The first chart by default shows the efficient frontier calculated by returns in the past. Additionally,
it shows the global minimum variance, equally risk contribution, equally weighted and maximum sharpe ratio portfolio. Changing the
direction of the toggle, the chart switches to showing the weights for the chosen portfolio.
The chart on the right shows the course of the price over the chosen period. Additionally, specific studies can be performed, for example
showing bollinger bands, moving averages and the relative strength of the portfolio.



## Screenshots

(screenshots/FirstView.png)
(screenshots/ERC_MA.png)


## Getting Started
For running the app locally, all the requirements have to be installed.


```
pip install -r requirements.txt

```

Run the app

```

python app.py

```

