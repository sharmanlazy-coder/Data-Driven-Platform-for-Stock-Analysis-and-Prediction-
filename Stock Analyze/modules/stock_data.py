import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json

class StockDataFetcher:
    def __init__(self):
        self.nse_stocks = {
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'HDFCBANK': 'HDFCBANK.NS',
            'INFY': 'INFY.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'HINDUNILVR': 'HINDUNILVR.NS',
            'ITC': 'ITC.NS',
            'SBIN': 'SBIN.NS',
            'BHARTIARTL': 'BHARTIARTL.NS',
            'KOTAKBANK': 'KOTAKBANK.NS',
            'LT': 'LT.NS',
            'AXISBANK': 'AXISBANK.NS',
            'ASIANPAINT': 'ASIANPAINT.NS',
            'MARUTI': 'MARUTI.NS',
            'SUNPHARMA': 'SUNPHARMA.NS',
            'TITAN': 'TITAN.NS',
            'BAJFINANCE': 'BAJFINANCE.NS',
            'WIPRO': 'WIPRO.NS',
            'HCLTECH': 'HCLTECH.NS',
            'ULTRACEMCO': 'ULTRACEMCO.NS',
            'ADANIENT': 'ADANIENT.NS',
            'TATAMOTORS': 'TATAMOTORS.NS',
            'TATASTEEL': 'TATASTEEL.NS',
            'POWERGRID': 'POWERGRID.NS',
            'ONGC': 'ONGC.NS',
            'NTPC': 'NTPC.NS',
            'JSWSTEEL': 'JSWSTEEL.NS',
            'COALINDIA': 'COALINDIA.NS',
            'BAJAJFINSV': 'BAJAJFINSV.NS',
            'TECHM': 'TECHM.NS',
        }
        self.cache = {}
        self.cache_timeout = 300
    
    def search_indian_stocks(self, query):
        if not query:
            return [{'name': name, 'symbol': symbol} for name, symbol in list(self.nse_stocks.items())[:10]]
        
        results = []
        for name, symbol in self.nse_stocks.items():
            if query.upper() in name.upper() or query.upper() in symbol.upper():
                results.append({'name': name, 'symbol': symbol})
        
        return results[:10]
    
    def get_stock_data(self, symbol):
        try:
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = f"{symbol}.NS"
            
            stock = yf.Ticker(symbol)
            info = stock.info
            
            if not info or 'currentPrice' not in info and 'regularMarketPrice' not in info:
                return {'error': f'Unable to fetch data for {symbol}. Please check the stock symbol and try again.'}
            
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            
            previous_close = info.get('previousClose', 0)
            if previous_close == 0:
                previous_close = info.get('regularMarketPreviousClose', 0)

            data = {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', symbol)),
                'current_price': round(current_price, 2),
                'currency': '₹',
                'open': round(info.get('open', info.get('regularMarketOpen', 0)), 2),
                'high': round(info.get('dayHigh', info.get('regularMarketDayHigh', 0)), 2),
                'low': round(info.get('dayLow', info.get('regularMarketDayLow', 0)), 2),
                'previous_close': round(previous_close, 2),
                'volume': info.get('volume', info.get('regularMarketVolume', 0)),
                'market_cap': info.get('marketCap', info.get('regularMarketCap', 0)),
                'pe_ratio': info.get('trailingPE', info.get('forwardPE', None)),
                '52_week_high': round(info.get('fiftyTwoWeekHigh', 0), 2),
                '52_week_low': round(info.get('fiftyTwoWeekLow', 0), 2),
            }

            # Calculate change and change percent
            if previous_close > 0:
                data['change'] = round(current_price - previous_close, 2)
                data['change_percent'] = round(((current_price - previous_close) / previous_close) * 100, 2)
            else:
                data['change'] = 0
                data['change_percent'] = 0
            
            return data
        except Exception as e:
            return {'error': 'Failed to load stock data. Please check your internet connection and try again.'}
    
    def get_historical_data(self, symbol, period='6mo'):
        try:
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = f"{symbol}.NS"
            
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            
            if hist.empty:
                return {'error': f'No historical data available for {symbol}. Please check the stock symbol.'}
            
            data = {
                'dates': hist.index.strftime('%Y-%m-%d').tolist(),
                'open': hist['Open'].round(2).tolist(),
                'high': hist['High'].round(2).tolist(),
                'low': hist['Low'].round(2).tolist(),
                'close': hist['Close'].round(2).tolist(),
                'volume': hist['Volume'].tolist(),
            }
            
            return data
        except Exception as e:
            return {'error': f'Failed to retrieve historical data for {symbol}. Please check your connection and try again.'}
    
    def get_nifty_data(self):
        try:
            nifty = yf.Ticker('^NSEI')
            hist = nifty.history(period='1y')
            
            if hist.empty:
                return None
            
            return hist['Close'].pct_change().mean() * 252
        except:
            return 0.12
