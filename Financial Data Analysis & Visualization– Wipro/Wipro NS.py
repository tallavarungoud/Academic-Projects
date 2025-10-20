import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (14,8)

def download_price(ticker, period='5y', interval='1d'):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if df is None or df.empty:
        raise ValueError(f"No price data returned for {ticker}")
        
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

def get_fundamentals(ticker):
    tk = yf.Ticker(ticker)
    pnl = tk.financials.T
    bs = tk.balance_sheet.T
    cf = tk.cashflow.T
    
    fundamentals = pd.concat([pnl, bs, cf], axis=1)
    fundamentals.reset_index(inplace=True)
    fundamentals.rename(columns={'index':'Date'}, inplace=True)
    return fundamentals

def save_fundamentals_table(df, filename='fundamentals_concatenated.csv'):
    if df is None or df.empty:
        return
    df.to_csv(filename, index=False)

def SMA(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def EMA(series, window):
    return series.ewm(span=window, adjust=False).mean()

def RSI(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def plot_price_with_indicators(price_df, title=None):
    df = price_df.copy()

    if 'Adj Close' in df.columns:
        price = df['Adj Close']
    else:
        price = df['Close']

    df['SMA50'] = SMA(price, 50)
    df['SMA200'] = SMA(price, 200)
    df['EMA20'] = EMA(price, 20)
    df['RSI14'] = RSI(price, 14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = MACD(price)

    fig, axs = plt.subplots(4, 1, sharex=True, 
                            gridspec_kw={'height_ratios': [3, 1, 1, 1]}, 
                            figsize=(14, 10))
    
    ax_price, ax_vol, ax_macd, ax_rsi = axs
    
    x_coords_num = np.arange(len(df.index)) 
    
    # 1. Price Panel
    ax_price.plot(x_coords_num, price, label='Adj Close')
    ax_price.plot(x_coords_num, df['SMA50'], label='SMA50')
    ax_price.plot(x_coords_num, df['SMA200'], label='SMA200')
    ax_price.plot(x_coords_num, df['EMA20'], label='EMA20')
    ax_price.legend(loc='upper left')
    ax_price.grid(True, linestyle='--', alpha=0.6)
    ax_price.set_ylabel('Price')
    
    # 2. Volume Panel
    if 'Volume' in df.columns:
        ax_vol.vlines(x_coords_num, 0, df['Volume'].values, color='gray', linewidth=0.8) 
        ax_vol.set_ylabel('Volume')
        ax_vol.grid(True, linestyle='--', alpha=0.6)

    # 3. MACD Panel
    ax_macd.plot(x_coords_num, df['MACD'], label='MACD', color='blue')
    ax_macd.plot(x_coords_num, df['MACD_signal'], label='Signal', color='red')
    
    zero_line = [0] * len(x_coords_num)
    ax_macd.vlines(x_coords_num, zero_line, df['MACD_hist'].values, 
                   color=np.where(df['MACD_hist'].values > 0, 'green', 'red'), 
                   linewidth=1.0, label='Histogram')
    
    ax_macd.legend(loc='upper left')
    ax_macd.set_ylabel('MACD')
    ax_macd.grid(True, linestyle='--', alpha=0.6)

    # 4. RSI Panel
    ax_rsi.plot(x_coords_num, df['RSI14'], label='RSI14', color='purple')
    ax_rsi.axhline(70, linestyle='--', color='red', alpha=0.7)
    ax_rsi.axhline(30, linestyle='--', color='green', alpha=0.7)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel('RSI')
    ax_rsi.grid(True, linestyle='--', alpha=0.6)
    
    tick_spacing = 60 
    tick_locations = x_coords_num[::tick_spacing]
    ax_rsi.set_xticks(tick_locations) 
    tick_labels = df.index.strftime('%Y-%m-%d')[::tick_spacing]
    ax_rsi.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    ax_price.tick_params(axis='x', labelbottom=False)
    ax_vol.tick_params(axis='x', labelbottom=False)
    ax_macd.tick_params(axis='x', labelbottom=False)

    plt.suptitle(title or 'Price & Indicators', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show() 


# Change the company name to Wipro
TICKER_NAME = 'WIPRO.NS'

# 1. Collect Price Data
df = download_price(TICKER_NAME, period='5y', interval='1d')

# 2. Collect and Save Fundamentals
fundamentals_df = get_fundamentals(TICKER_NAME)
save_fundamentals_table(fundamentals_df)

# Save to the new directory
output_path = r"C:\Users\Admin\OneDrive\Desktop\DF PROJECT\fundamentals_wipro.csv"
fundamentals_df.to_csv(output_path, index=False)
print(f"Fundamentals saved successfully to: {output_path}")  
