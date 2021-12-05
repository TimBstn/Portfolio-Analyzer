# standard libraries
import pandas as pd
from datetime import date
import random
from datetime import timedelta

# dash and plotly
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import State, Input, Output
import dash_daq as daq
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

# finance libraries
import finance as erk
import bt

# css for pictograms
FONT_AWESOME = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)

app.title = "Portfolio Performance Analyser"

# This is for gunicorn
server = app.server

# Side panel

# dropdown for weighting type
type_dropdown = dcc.Dropdown(
    id="type-dropdown-component",
    options=[
        {'label': 'Equally Weighted', 'value': 'EW'},
        {'label': 'Inverse Volatility', 'value': 'INV'},
        {'label': 'Global Minimum Volatility', 'value': 'GMV'},
        {'label': 'Equal Risk Contribution', 'value': 'ERC'},
        {'label': 'Randomly', 'value': 'RANDOM'},
    ],
    value='EW',
    clearable=False,
    optionHeight=45,
)

# dropdown for rebalance period
rebalance_dropdown = dcc.Dropdown(
    id="rebalance-dropdown-component",
    options=[
        {'label': 'Daily', 'value': 'daily'},
        {'label': 'Weekly', 'value': 'weekly'},
        {'label': 'Monthly', 'value': 'monthly'},
        {'label': 'Quarterly', 'value': 'quarterly'},
        {'label': 'Yearly', 'value': 'yearly'},
        {'label': 'Once', 'value': 'once'},
    ],
    value='monthly',
    clearable=False,
)

# dropdown for all the nasdaq stocks
nasdaq_dropdown = dcc.Dropdown(
    id="nasdaq-dropdown-component",
    options=[
        {'label': stock, 'value': erk.NASDAQ100[stock]} for stock in erk.NASDAQ100
    ],
    clearable=False,
    multi=True,
    optionHeight=45,
)

# dropdown for all the dax stocks
dax_dropdown = dcc.Dropdown(
    id="dax-dropdown-component",
    options=[
        {'label': stock, 'value': erk.DAX[stock]} for stock in erk.DAX
    ],
    clearable=False,
    multi=True,
    optionHeight=45,
)

# side panel header
portfolio_dropdown_text = html.P(
    id="portfolio-dropdown-text", children=["Portfolio", html.Br(), " Construction"]
)

# titles for all the dropdowns
nasdaq_title = html.H1(id="nasdaq-name", children="NASDAQ 100")
dax_title = html.H1(id="dax-name", children="DAX 40")
time_title = html.H1(id="time-name", children="Start Date")
rebalance_title = html.H1(id="rebalance-name", children="Rebalance Period")

# calender for picking the start date
start_date = dcc.DatePickerSingle(
    id='start_date_picker',
    min_date_allowed=date(2005, 1, 1),
    max_date_allowed=date(2020, 1, 1),
    initial_visible_month=date(200, 1, 1),
    date=date(2010, 1, 1)
)

# style for submit button
button_style = {'background-color': '#2b2b2b',
                'color': 'white',
                'height': '40px',
                'width': '190px',
                'margin-top': '30px',
                'font-weight': 'bold',
                'border-color': 'white'
                }

# submit button to start calculation
submit_button = dbc.Button('Calculate', id='submit-val', n_clicks=0, style=button_style)

# tooltip when you hover over the submit button
tooltip_button = dbc.Tooltip('Start backtest, this might take a second!',
                             target='submit-val',
                             placement='top',
                             )

# bringing the side panel together
side_panel_layout = html.Div(
    id="panel-side",
    children=[
        portfolio_dropdown_text,
        html.Div(id="type-dropdown", children=[type_dropdown]),
        html.Div(id="rebalance_title", children=[rebalance_title]),
        html.Div(id="rebalance-dropdown", children=[rebalance_dropdown]),
        html.Div(id="nasdaq_title", children=[nasdaq_title]),
        html.Div(id="nasdaq-dropdown", children=[nasdaq_dropdown]),
        html.Div(id="dax_title", children=[dax_title]),
        html.Div(id="dax-dropdown", children=[dax_dropdown]),
        html.Div(id="time_title", children=[time_title]),
        html.Div(id="start_date", children=[start_date]),
        html.Div(id="calc_button", children=[submit_button, tooltip_button]),
    ],
)

# main panel

# cards

# style for the pictogram card
card_icon = {
    "color": "white",
    "textAlign": "center",
    "fontSize": 30,
    "margin": "auto",
}

# two cards next to each other, first with the number, second with the pictogram
card_ann_return = dbc.CardGroup(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5(id='ann_return_card',
                            children="0%", className="card-title",
                            style={'color': '#000', 'font-weight': 'bold', 'font-size': 23}),
                    html.P("Yearly Returns", className="card-text", style={'color': '#000'}),
                ]
            ),
            style={'background-image': 'linear-gradient(to right, #F9E8B4, #fec036)'},
        ),
        dbc.Card(
            html.Div(className="fa fa-line-chart", style=card_icon),
            style={"maxWidth": 75, "background": "#fec036"},
        ),
    ],
    className="mt-4 shadow",
)

card_ann_vol = dbc.CardGroup(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5(
                        id='ann_vol_card',
                        children="0%", className="card-title",
                        style={'color': '#000', 'font-weight': 'bold', 'font-size': 23}),
                    html.P("Yearly Volatility", className="card-text", style={'color': '#000'}),
                ]
            ),
            style={'background-image': 'linear-gradient(to right, #F9E8B4, #fec036)'},
        ),
        dbc.Card(
            html.Div(className="fa fa-area-chart", style=card_icon),
            style={"maxWidth": 75, "background": "#fec036"},
        ),
    ],
    className="mt-4 shadow",
)

card_max_drawdown = dbc.CardGroup(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5(
                        id="max_drawdown",
                        children="10%", className="card-title",
                        style={'color': '#000', 'font-weight': 'bold', 'font-size': 23}),
                    html.P("Max Drawdown", className="card-text", style={'color': '#000'}),
                ]
            ),
            style={'background-image': 'linear-gradient(to right, #F9E8B4, #fec036)'},
        ),
        dbc.Card(
            html.Div(className="fa fa-arrow-down", style=card_icon),
            style={"maxWidth": 75, "background": "#fec036"},
        ),
    ],
    className="mt-4 shadow",
)

card_sharpe = dbc.CardGroup(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5(
                        id="ann_sharpe",
                        children="10%", className="card-title",
                        style={'color': '#000', 'font-weight': 'bold', 'font-size': 23}),
                    html.P("Sharpe Ratio", className="card-text", style={'color': '#000'}),
                ]
            ),
            style={'background-image': 'linear-gradient(to right, #F9E8B4, #fec036)'},
        ),
        dbc.Card(
            html.Div(className="fa fa-star-half-o", style=card_icon),
            style={"maxWidth": 75, "background": "#fec036"},
        ),
    ],
    className="mt-4 shadow",
)

# create layer for all cards next to each other
cards = dbc.Row(
    [
        dbc.Col(card_ann_return, width=3),
        dbc.Col(card_ann_vol, width=3),
        dbc.Col(card_sharpe, width=3),
        dbc.Col(card_max_drawdown, width=3),
    ]
)

# charts

# price graph
# checklist for available studies
chart_checklist = dcc.Dropdown(
    id="chart_dropdown-component",
    options=[
        {'label': 'Bollinger Bands', 'value': 'BB'},
        {'label': 'Expo Weighted Moving Avg', 'value': 'EMA'},
        {'label': 'Simple Moving Avg', 'value': 'SMA'},
        {'label': 'Relative Strength Index', 'value': 'RSI'},  # with line at 70 and 30, second axis?
        {'label': 'Price Rate of Change', 'value': 'ROC'},

    ],
    optionHeight=45,
    placeholder="Studies",
)

# graph on the right with price data, studies checklist
price_graph = html.Div(
    id="price-container",
    className="six columns",
    children=[
        html.Div(
            id="price-header",
            children=[
                html.H1(
                    id="price-title", children=["Portfolio Performance"]
                ),
                chart_checklist
            ],
        ),
        dcc.Graph(
            id="price-graph",
            style={"width": 500, "margin": 0, 'display': 'inline-block'},
            config={"displayModeBar": False, "scrollZoom": False},

        ),
    ],
)

# weights graph on the left
# toggle for deciding if frontier or weights should be shown
weight_toggle = daq.ToggleSwitch(
    id="control-panel-toggle-weight",
    value=True,
    label=["Weights", "Effective Frontier"],
    style={"color": "#black"},
)

# diagram with toggle and graph
weight_diagram = html.Div(
    id="weight-container",
    className="six columns",
    children=[
        html.Div(
            id="weight-header",
            children=[
                html.H1(
                    id="weight-title", children=["Portfolio Properties"]
                ),
                weight_toggle,
            ],
        ),
        dcc.Graph(
            id="weight-graph",
            style={"width": 500, "margin": 0, 'display': 'inline-block'},
            config={"displayModeBar": False, "scrollZoom": False},
        )
    ],
)

# bringing the main panel together
main_panel_layout = html.Div(
    id="panel-upper-lower",
    children=[
        cards,
        html.Div(
            id="charts",
            className="row",
            children=[
                weight_diagram, price_graph
            ]
        )
    ],
)

# bringing everything together, creating store-divs for used data
root_layout = html.Div(
    id="root",
    children=[
        dcc.Store(
            id="store-data",
            data={}
        ),
        dcc.Store(
            id="store-backtests-stats",
            data={}
        ),
        dcc.Store(
            id="store-backtests-prices",
            data={}
        ),
        dcc.Store(
            id="store-backtests-weights",
            data={}
        ),
        side_panel_layout,
        main_panel_layout,
    ],
)

# creating the app
app.layout = root_layout


# Callbacks
# Callbacks for creating data, store it in Store-div

# get price data
@app.callback(
    [
        Output('store-data', 'data'),
    ],
    [
        Input('submit-val', 'n_clicks')
    ],
    [
        State('nasdaq-dropdown-component', 'value'),
        State('dax-dropdown-component', 'value'),
        State('start_date_picker', 'date'),
        State("store-data", "data"),
    ],

)
def get_data(n_clicks, nasdaq_stocks, dax_stocks, start_date_stocks, data):
    """
    :return: starting with clicking the Calculate button, the function returns the price data for the selected stocks
    and saves it into the price-store
    """
    if nasdaq_stocks is None:
        nasdaq_stocks = []
    if dax_stocks is None:
        dax_stocks = []

    # all the selected stocks
    all_stocks = nasdaq_stocks + dax_stocks

    # pull the data
    return_data = erk.get_returns(all_stocks, start_date=start_date_stocks)

    if return_data.empty:
        return [data]

    # return in json format
    return [return_data.to_json(date_format='iso', orient='split')]


# create the backtest data, save stats, prices and weights
@app.callback(
    [
        Output('store-backtests-stats', 'data'),
        Output('store-backtests-prices', 'data'),
        Output('store-backtests-weights', 'data'),
    ],
    [
        Input('submit-val', 'n_clicks'),
        Input('store-data', 'data')
    ],
    [
        State('rebalance-dropdown-component', 'value'),
        State("store-backtests-stats", "data"),
        State("store-backtests-prices", "data"),
        State("store-backtests-weights", "data"),
    ],

)
def get_backtest(n_clicks, stored_data, rebalance_period, data_stats, data_prices, data_weights):
    """
    :return: Starting with the calculate button, the function creates a backtest for every strategy and saves the stats,
    price data and weights of that strategy
    """
    if stored_data == {}:
        return data_stats, data_prices, data_weights

    # converting price data into dataframe
    df = pd.read_json(stored_data, orient='split')

    # generating the weights for erc and gmv portfolios
    erc_weights, gmv_weights = erk.get_weights_erc_gmv(df, rebalance_period=rebalance_period)

    if rebalance_period == "daily":
        period = bt.algos.RunDaily()
    elif rebalance_period == "weekly":
        period = bt.algos.RunWeekly()
    elif rebalance_period == "monthly":
        period = bt.algos.RunMonthly()
    elif rebalance_period == "quarterly":
        period = bt.algos.RunQuarterly()
    elif rebalance_period == "yearly":
        period = bt.algos.RunYearly()
    else:
        period = bt.algos.RunOnce()

    # run backtests
    ew = bt.Strategy("EW", [period,
                            bt.algos.SelectAll(),
                            bt.algos.WeighEqually(),
                            bt.algos.Rebalance()])
    backtest_ew = bt.Backtest(ew, df)

    inv = bt.Strategy("INV", [period,
                              bt.algos.SelectAll(),
                              bt.algos.WeighInvVol(),
                              bt.algos.Rebalance()])
    backtest_inv = bt.Backtest(inv, df)

    erc = bt.Strategy("ERC", [period,
                              bt.algos.SelectAll(),
                              bt.algos.WeighTarget(erc_weights),
                              bt.algos.Rebalance()])
    backtest_erc = bt.Backtest(erc, df)

    gmv = bt.Strategy("GMV", [period,
                              bt.algos.SelectAll(),
                              bt.algos.WeighTarget(gmv_weights),
                              bt.algos.Rebalance()])
    backtest_gmv = bt.Backtest(gmv, df)

    ran = bt.Strategy("RANDOM", [period,
                                 bt.algos.SelectAll(),
                                 bt.algos.WeighRandomly(),
                                 bt.algos.Rebalance()])
    backtest_ran = bt.Backtest(ran, df)

    res = bt.run(backtest_ew, backtest_inv, backtest_erc, backtest_gmv, backtest_ran)

    # create stats for backtests
    stats = res.stats

    # save prices for each strategy in dataframe
    prices = pd.DataFrame({'EW': res.backtests['EW'].stats.prices, 'INV': res.backtests['INV'].stats.prices,
                           'ERC': res.backtests['ERC'].stats.prices, 'GMV': res.backtests['GMV'].stats.prices,

                           'RANDOM': res.backtests['RANDOM'].stats.prices})

    # save weights for each strategy in dataframe
    erc_weights.columns = ["ERC>" + col for col in erc_weights]
    gmv_weights.columns = ["GMV>" + col for col in gmv_weights]

    weights = pd.concat([res.backtests['EW'].weights, res.backtests['INV'].weights,
                         res.backtests['ERC'].weights, res.backtests['GMV'].weights,
                         res.backtests['RANDOM'].weights], axis=1)

    return stats.to_json(date_format='iso', orient='split'), \
           prices.to_json(date_format='iso', orient='split'), \
           weights.to_json(date_format='iso', orient='split')


# Callback Cards
@app.callback(
    [
        Output('ann_return_card', 'children'),
        Output('ann_vol_card', 'children'),
        Output('ann_sharpe', 'children'),
        Output('max_drawdown', 'children'),
    ],
    [
        Input('store-backtests-stats', 'data'),
        Input('type-dropdown-component', 'value'),
    ]
)
def update_card(data, type_portfolio):
    """
    :return: taking the stats dataframe created in the function above, this function updates the card values
    """
    if data == {}:
        return ["0%"], ["0"], ["0"], ["0%"]

    # convert json to dataframe
    df = pd.read_json(data, orient='split')

    # update cards
    ann = round(df.loc['cagr', type_portfolio] * 100, 1)
    vol = round(df.loc['yearly_vol', type_portfolio], 2)
    sharpe = round(df.loc['yearly_sharpe', type_portfolio], 2)
    draw = round(df.loc['max_drawdown', type_portfolio] * 100, 1)
    return [f"{ann}%"], [f"{vol}"], [f"{sharpe}"], [f"{draw}%"]


# callback graphs
# updating graph on the left
@app.callback(
    [
        Output('weight-graph', 'figure'),
    ],
    [
        Input('submit-val', 'n_clicks'),
        Input('type-dropdown-component', 'value'),
        Input('control-panel-toggle-weight', 'value'),
        Input('store-backtests-weights', 'data'),
        Input('store-data', 'data'),
    ],
    [
        State('rebalance-dropdown-component', 'value'),

    ]

)
def update_weight_graph(n_clicks, type_weighting, chart_type, data_weights, data_ef, rebalance_period):

    # general graph layout
    layout = go.Layout(
        {
            "margin": {"t": 30, "r": 35, "b": 40, "l": 50},
            "xaxis": {"showgrid": False},
            "yaxis": {"showgrid": False},
            "plot_bgcolor": "#2b2b2b",
            "paper_bgcolor": "#2b2b2b",
            "font": {"color": "gray"},
        }
    )

    # colors for the lines
    colors = px.colors.sequential.turbid + px.colors.sequential.Oranges + px.colors.sequential.YlOrBr
    random.Random(4).shuffle(colors)

    if data_weights == {}:
        d = go.Scatter(x=[1, 2, 3, 4], y=[1, 2, 3, 4], line=dict(color='#fec036'))
        return [go.Figure(data=d, layout=layout)]

    # read in data and convert to dataframe
    df = pd.read_json(data_weights, orient='split')
    df_ef = pd.read_json(data_ef, orient='split')

    # show different time period in weight graph for every rebalance period chosen
    if rebalance_period == "daily":
        df = df
        df = df.tail(n=30)
    elif rebalance_period == "weekly":
        df = df.groupby([df.index.year, df.index.week]).head(1)
        df = df.tail(n=25)
    elif rebalance_period == "monthly":
        df = df.groupby([df.index.year, df.index.month]).head(1)
        df = df.tail(n=12)
    elif rebalance_period == "quarterly":
        df = df.groupby([df.index.year, df.index.quarter]).head(1)
        df = df.tail(n=12)
    elif rebalance_period == "yearly":
        df = df.groupby([df.index.year, df.index.year]).head(1)
    else:
        df = df.groupby([df.index.year, df.index.month]).head(1)
        df = df.tail(n=12)

    # columns come in format type>stock, filter for the weighting type selected
    filter_col = [col for col in df if col.startswith(type_weighting + ">")]

    # filter for type
    df = df[filter_col]

    # translate columns into stocks only
    cols = [col.split(">", 1)[1] for col in df]
    df.columns = cols
    df = df.loc[~(df == 0).all(axis=1)]

    # toggle on effective Frontier
    if chart_type:
        fig = erk.plot_ef(df_ef, layout)
        fig.update_layout(yaxis_tickformat='.2%')

    # toggle on weights
    else:
        data = []
        i = 0
        # create line for every stock
        for stock in df:
            trace = go.Scatter(x=df.index,
                               y=df[stock],
                               name=stock,
                               line=dict(color=colors[i]))
            data.append(trace)
            i += 1

        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(yaxis_tickformat='.2%')
    return [fig]


# callback for price chart on the right
@app.callback(
    [
        Output('price-graph', 'figure')
    ],
    [
        Input('type-dropdown-component', 'value'),
        Input('store-backtests-prices', 'data'),
        Input('chart_dropdown-component', 'value'),
    ]
)
def update_price_graph(type_weighting, data_price, study):

    # general layout for graph
    layout = {
        "margin": {"t": 30, "r": 35, "b": 40, "l": 50},
        "xaxis": {"showgrid": False},
        "yaxis": {"showgrid": False},
        "plot_bgcolor": "#2b2b2b",
        "paper_bgcolor": "#2b2b2b",
        "font": {"color": "gray"},
    }

    if data_price == {}:
        data = []
        trace = go.Scatter(x=[1, 2, 3, 4], y=[1, 2, 3, 4], line=dict(color='#fec036'))
        data.append(trace)
        fig = go.Figure(data=data, layout=layout)
        return [fig]

    # read in data and convert to dataframe
    df_price = pd.read_json(data_price, orient='split')

    # create trace for price
    data = []
    trace_price = go.Scatter(x=df_price.index, y=df_price[type_weighting], line=dict(color='#fec036'), showlegend=False)
    data.append(trace_price)

    # no study selected
    if study is None or study == []:
        fig = go.Figure(data=data, layout=layout)
        return [fig]

    # EMA selected
    if 'EMA' in study:
        # create expo weighted moving average for lookback periods 20 and 5
        df_price['EWMA20'] = df_price[type_weighting].ewm(span=20, adjust=False).mean()
        df_price['EWMA5'] = df_price[type_weighting].ewm(span=5, adjust=False).mean()

        # create traces for both lines
        trace_ema20 = go.Scatter(
            x=df_price.index, y=df_price["EWMA20"], mode="lines", name="EMA20", line=dict(color='#9BC8F0')
        )
        data.append(trace_ema20)

        trace_ema5 = go.Scatter(
            x=df_price.index, y=df_price["EWMA5"], mode="lines", name="EMA5", line=dict(color='#0588FD')
        )
        data.append(trace_ema5)

        # change layout x axis --> only show last 4 month
        layout['xaxis_range'] = [max(df_price.index) - timedelta(days=30 * 6), max(df_price.index)]

        # change layout y axis --> only show data between min and max of columns
        layout['yaxis_range'] = [df_price.loc[max(df_price.index) - timedelta(days=30 * 6): max(df_price.index),
                                 [type_weighting, 'EWMA20', 'EWMA5']].values.min() - 20,
                                 df_price.loc[max(df_price.index) - timedelta(days=30 * 6): max(df_price.index),
                                 [type_weighting, 'EWMA20', 'EWMA5']].values.max() + 20]

        fig = go.Figure(data=data, layout=layout)
        return [fig]

    # SMA selected
    if 'SMA' in study:
        # create simple moving average for lookback periods 20 and 5
        df_price['SMA20'] = df_price[type_weighting].rolling(window=20).mean()
        df_price['SMA5'] = df_price[type_weighting].rolling(window=5).mean()

        # create traces for both lines
        trace_sma20 = go.Scatter(
            x=df_price.index, y=df_price["SMA20"], mode="lines", name="SMA20", line=dict(color='#F7B483')
        )
        data.append(trace_sma20)

        trace_sma5 = go.Scatter(
            x=df_price.index, y=df_price["SMA5"], mode="lines", name="SMA5", line=dict(color='#CD7637')
        )
        data.append(trace_sma5)

        # change layout x axis --> only show last 6 month
        layout['xaxis_range'] = [max(df_price.index) - timedelta(days=30 * 6), max(df_price.index)]

        # change layout y axis --> only show data between min and max of columns
        layout['yaxis_range'] = [df_price.loc[max(df_price.index) - timedelta(days=30 * 6): max(df_price.index),
                                 [type_weighting, 'SMA20', 'SMA5']].values.min() - 20,
                                 df_price.loc[max(df_price.index) - timedelta(days=30 * 6): max(df_price.index),
                                 [type_weighting, 'SMA20', 'SMA5']].values.max() + 20]

        fig = go.Figure(data=data, layout=layout)
        return [fig]

    # Bollinger Bands selected
    if 'BB' in study:
        # create column for rolling mean and standard deviation (20 days lookback), upper and lower band (3 stds away
        # from mean)
        df_price['Rolling_Mean'] = df_price[type_weighting].rolling(window=20).mean()
        df_price['Rolling_Std'] = df_price[type_weighting].rolling(window=20).std()
        df_price['Upper_Band'] = df_price['Rolling_Mean'] + 3 * df_price['Rolling_Std']
        df_price['Lower_Band'] = df_price['Rolling_Mean'] - 3 * df_price['Rolling_Std']

        # create trace for the 3 lines
        trace_upper = go.Scatter(
            x=df_price.index, y=df_price['Upper_Band'], mode="lines", name="BB Upper", line=dict(color='#70C01A')
        )

        # fill area between upper and lower band --> tonexty
        trace_lower = go.Scatter(
            x=df_price.index, y=df_price['Lower_Band'], mode="lines", name="BB Lower", line=dict(color='#BBF77A'),
            fill='tonexty', fillcolor='rgba(214,250,175,0.3)'
        )

        trace_mean = go.Scatter(
            x=df_price.index, y=df_price['Rolling_Mean'], mode="lines", showlegend=False, name="BB_mean",
            line=dict(color='#A8F456')
        )

        data.append(trace_mean)
        data.append(trace_upper)
        data.append(trace_lower)

        # change layout x axis --> only show last 6 month
        layout['xaxis_range'] = [max(df_price.index) - timedelta(days=30 * 6), max(df_price.index)]

        # change layout y axis --> only show data between min and max of columns
        layout['yaxis_range'] = [df_price.loc[max(df_price.index) - timedelta(days=30 * 6): max(df_price.index),
                                 [type_weighting, 'Lower_Band']].values.min() - 20,
                                 df_price.loc[max(df_price.index) - timedelta(days=30 * 6): max(df_price.index),
                                 [type_weighting, 'Upper_Band']].values.max() + 20]

        fig = go.Figure(data=data, layout=layout)
        return [fig]

    # Price Return of Capital selected
    if 'ROC' in study:
        # create columns for price 20 days ago and difference for price today and 20 days ago
        df_price['N'] = df_price[type_weighting].diff(20)
        df_price['D'] = df_price[type_weighting].shift(20)
        # calculate ROC
        df_price['ROC'] = pd.Series(df_price['N'] / df_price['D'])

        # layout for roc graph
        layout_roc = {
            'plot_bgcolor': "#2b2b2b",
            'paper_bgcolor': "#2b2b2b",
            "xaxis": {"showgrid": False},
            "xaxis2": {"showgrid": False},
            "yaxis": {"showgrid": False},
            "yaxis2": {"showgrid": False},
            "margin": {"t": 10, "r": 35, "b": 10, "l": 50},
            "font": {"color": "gray"},
        }

        # create two plots: roc on the bottom, price on the top
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

        # add price trace
        fig.add_trace(
            trace_price,
            row=1, col=1,
        )

        # add roc trace
        fig.add_trace(
            go.Scatter(x=df_price.index, y=df_price['ROC'], mode="lines", showlegend=False, name="ROC",
                       line=dict(color='#EF752B'), ),
            row=2, col=1,

        )

        fig.update_layout(layout_roc)
        return [fig]

    # Relative Strength Index is selected
    if 'RSI' in study:
        # calculate price difference to day before
        df_price['diff'] = df_price[type_weighting].diff()

        # Make two series: one for lower closes and one for higher closes
        df_price['up'] = df_price['diff'].clip(lower=0)
        df_price['down'] = -1 * df_price['diff'].clip(upper=0)

        # Use exponential moving average
        df_price['ma_up'] = df_price['up'].ewm(com=14 - 1, adjust=True, min_periods=14).mean()
        df_price['ma_down'] = df_price['down'].ewm(com=14 - 1, adjust=True, min_periods=14).mean()

        # calculate rsi
        df_price['rsi'] = df_price['ma_up'] / df_price['ma_down']
        df_price['rsi'] = 100 - (100 / (1 + df_price['rsi']))

        # layout for rsi graph
        layout_rsi = {
            'plot_bgcolor': "#2b2b2b",
            'paper_bgcolor': "#2b2b2b",
            "xaxis": {"showgrid": False},
            "xaxis2": {"showgrid": False},
            "yaxis": {"showgrid": False},
            "yaxis2": {"showgrid": False},
            'yaxis2_range': [0, 100],
            "margin": {"t": 10, "r": 35, "b": 10, "l": 50},
            "font": {"color": "gray"},
        }

        # create two plots: rsi on the bottom, price on the top
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

        # trace for price data
        fig.add_trace(
            trace_price,
            row=1, col=1,
        )

        # trace for rsi
        fig.add_trace(
            go.Scatter(x=df_price.index, y=df_price['rsi'], mode="lines", showlegend=False, name="RSI",
                       line=dict(color='#EF752B'), ),
            row=2, col=1,

        )

        fig.update_layout(layout_rsi)

        # add horizontal lines at 30 and 70
        fig['layout'].update(shapes=[{'type': 'line', 'y0': 70, 'y1': 70, 'x0': str(df_price.index[0]),
                                      'x1': str(df_price.index[-1]), 'xref': 'x2', 'yref': 'y2',
                                      'line': {'color': '#FFF', 'width': 2.5, 'dash': 'dash'}},
                                     {'type': 'line', 'y0': 30, 'y1': 30, 'x0': str(df_price.index[0]),
                                      'x1': str(df_price.index[-1]), 'xref': 'x2', 'yref': 'y2',
                                      'line': {'color': '#FFF', 'width': 2.5, 'dash': 'dash'}}
                                     ])

        return [fig]


if __name__ == "__main__":
    app.run_server(debug=False)
app.scripts.config.serve_locally = True