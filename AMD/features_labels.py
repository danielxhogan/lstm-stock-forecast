# AMD, NVDA, INTC, AND XLK TIMESERIES FORECASTING
# *****************************************************************************
import datetime
import random
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from sklearn import preprocessing
from collections import deque
import talib as ta
from statsmodels.tsa.arima_model import ARIMA

SEQ_LEN = 60
VAL_PCT = 0.1
FUTURE_PERIOD_PREDICT = 1
DAYS_OF_DATA = 5000
TICKER_TO_PREDICT = 'AMD'

tickers = ['AMD', 'NVDA', 'INTC', 'XLK', 'QQQ', 'SPY']

# *****************************************************************************
def get_data(ticker):
    ticker = ticker.upper()
    end_date = datetime.datetime.today()
    start_date = end_date-datetime.timedelta(days=DAYS_OF_DATA)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    df = pdr.DataReader(ticker,'yahoo',start_date,end_date)
    return df

# *****************************************************************************
def make_features_labels():
    X = pd.DataFrame()
    
    for ticker in tickers:
        data = get_data(ticker)
        data.drop(columns=['Adj Close'], inplace=True)
        
        for column in data.columns:
            X[f'{ticker}-{column}'] = data[column]
    
    X['tomorrow'] = X[f'{TICKER_TO_PREDICT}-Close'].shift(-FUTURE_PERIOD_PREDICT) > X[f'{TICKER_TO_PREDICT}-Close']
    X['tomorrow'] = X['tomorrow'] * 1
    
    return X

# *****************************************************************************
def calculate_technicals(df):
    # model = ARIMA(df[f'{TICKER_TO_PREDICT}-Close'].values, order=(5,1,0))
    # model_fit = model.fit(disp=0)
    # output = model_fit.forecast()
    
    df[f'{TICKER_TO_PREDICT}-50-ema'] = ta.EMA(df[f'{TICKER_TO_PREDICT}-Close'], timeperiod=50)
    df[f'{TICKER_TO_PREDICT}-100-ema'] = ta.EMA(df[f'{TICKER_TO_PREDICT}-Close'], timeperiod=100)
    df[f'{TICKER_TO_PREDICT}-200-ema'] = ta.EMA(df[f'{TICKER_TO_PREDICT}-Close'], timeperiod=200)
    df[f'{TICKER_TO_PREDICT}-obv'] = ta.OBV(df[f'{TICKER_TO_PREDICT}-Close'], df[f'{TICKER_TO_PREDICT}-Volume'])
    df[f'{TICKER_TO_PREDICT}-ad'] = ta.AD(df[f'{TICKER_TO_PREDICT}-High'],
                                          df[f'{TICKER_TO_PREDICT}-Low'],
                                          df[f'{TICKER_TO_PREDICT}-Close'],
                                          df[f'{TICKER_TO_PREDICT}-Volume'])
    df[f'{TICKER_TO_PREDICT}-adosc'] = ta.ADOSC(df[f'{TICKER_TO_PREDICT}-High'],
                                                df[f'{TICKER_TO_PREDICT}-Low'],
                                                df[f'{TICKER_TO_PREDICT}-Close'],
                                                df[f'{TICKER_TO_PREDICT}-Volume'])
    df[f'{TICKER_TO_PREDICT}-adx'] = ta.ADX(df[f'{TICKER_TO_PREDICT}-High'],
                                            df[f'{TICKER_TO_PREDICT}-Low'],
                                            df[f'{TICKER_TO_PREDICT}-Close'])
    (df[f'{TICKER_TO_PREDICT}-macd'],
    df[f'{TICKER_TO_PREDICT}-signal'],
    df[f'{TICKER_TO_PREDICT}-hist']) = ta.MACD(df[f'{TICKER_TO_PREDICT}-Close'])
    df[f'{TICKER_TO_PREDICT}-rsi'] = ta.RSI(df[f'{TICKER_TO_PREDICT}-Close'])
    df[f'{TICKER_TO_PREDICT}-atr'] = ta.ATR(df[f'{TICKER_TO_PREDICT}-High'],
                                            df[f'{TICKER_TO_PREDICT}-Low'],
                                            df[f'{TICKER_TO_PREDICT}-Close'])
    (df[f'{TICKER_TO_PREDICT}-upper'],
    df[f'{TICKER_TO_PREDICT}-middle'],
    df[f'{TICKER_TO_PREDICT}-lower']) = ta.BBANDS(df[f'{TICKER_TO_PREDICT}-Close'])
    
    df.dropna(inplace=True)
    
    return df

# *****************************************************************************
def make_train_test(X, pct):
    test_amt = int(len(X) * pct)
    
    train_X = X[:-test_amt]
    
    test_X = X[-test_amt:]
    
    return train_X, test_X

# *****************************************************************************
def preprocess_data(df):
    columns = [column for column in df.columns if column != 'tomorrow']
    
    for column in columns:
        print(column)
        df[column] = df[column].pct_change()
    print(df)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    
    print(df)
    
    for column in columns:
        df[column] =  preprocessing.scale(df[column].values)
        
    print(df)
        
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    features = df[columns]
    
    for i in range(len(features.values)):
        prev_days.append(features.values[i])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([ np.array(prev_days), df['tomorrow'][i] ])
            
    random.shuffle(sequential_data)
            
    return sequential_data

# *****************************************************************************
def balance_data(data):
    buys = []
    sells = []
    
    for features, label in data:
        if label == 1:
            buys.append([features, label])
        elif label == 0:
            sells.append([features, label])
    
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    
    data = buys + sells
    random.shuffle(data)
    
    return data

# *****************************************************************************
def seperate_features_labels(data):
    X = []
    y = []
    
    for features, label in data:
        X.append(features)
        y.append(label)
        
    return X, y


df = make_features_labels()
df = calculate_technicals(df)

train_df, test_df = make_train_test(df, VAL_PCT)

pre_train = preprocess_data(train_df)
balanced_train = balance_data(pre_train)

train_X, train_y = seperate_features_labels(balanced_train)

pre_test = preprocess_data(test_df)
balanced_test = balance_data(pre_test)

test_X, test_y = seperate_features_labels(balanced_test)


np.save('train_X.npy', np.array(train_X))
np.save('train_y.npy', np.array(train_y))
np.save('test_X.npy', np.array(test_X))
np.save('test_y.npy', np.array(test_y))


















