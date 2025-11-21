import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np

class TechnicalAnalyzer:
    def __init__(self):
        pass
    
    def get_all_indicators(self, symbol):
        try:
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = f"{symbol}.NS"
            
            stock = yf.Ticker(symbol)
            df = stock.history(period='6mo')
            
            if df.empty:
                return {'error': f'No historical data available for {symbol}. Please check the stock symbol and try again.'}
            
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            macd = ta.macd(df['Close'])
            if macd is not None:
                df['MACD'] = macd['MACD_12_26_9']
                df['MACD_signal'] = macd['MACDs_12_26_9']
                df['MACD_hist'] = macd['MACDh_12_26_9']
            
            bbands = ta.bbands(df['Close'], length=20)
            if bbands is not None and not bbands.empty:
                # Bollinger Bands columns vary based on pandas_ta version
                bb_cols = bbands.columns.tolist()
                upper_col = [c for c in bb_cols if 'BBU' in c][0] if any('BBU' in c for c in bb_cols) else None
                middle_col = [c for c in bb_cols if 'BBM' in c][0] if any('BBM' in c for c in bb_cols) else None
                lower_col = [c for c in bb_cols if 'BBL' in c][0] if any('BBL' in c for c in bb_cols) else None
                
                if upper_col:
                    df['BB_upper'] = bbands[upper_col]
                if middle_col:
                    df['BB_middle'] = bbands[middle_col]
                if lower_col:
                    df['BB_lower'] = bbands[lower_col]
            
            df['MA_20'] = ta.sma(df['Close'], length=20)
            df['MA_50'] = ta.sma(df['Close'], length=50)
            df['MA_100'] = ta.sma(df['Close'], length=100)
            
            df = df.dropna()
            
            latest_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else None
            latest_macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else None
            latest_signal = df['MACD_signal'].iloc[-1] if 'MACD_signal' in df.columns else None
            
            rsi_signal = 'Overbought' if latest_rsi and latest_rsi > 70 else 'Oversold' if latest_rsi and latest_rsi < 30 else 'Neutral'
            macd_signal = 'Bullish' if latest_macd and latest_signal and latest_macd > latest_signal else 'Bearish'
            
            dates_list = df.index.strftime('%Y-%m-%d').tolist()
            
            result = {
                'dates': dates_list,
                'close': df['Close'].round(2).tolist(),
                'rsi': {
                    'dates': dates_list,
                    'values': df['RSI'].round(2).tolist() if 'RSI' in df.columns else []
                },
                'rsi_latest': round(latest_rsi, 2) if latest_rsi else None,
                'rsi_signal': rsi_signal,
                'macd': {
                    'dates': dates_list,
                    'macd': df['MACD'].round(2).tolist() if 'MACD' in df.columns else [],
                    'signal': df['MACD_signal'].round(2).tolist() if 'MACD_signal' in df.columns else [],
                    'histogram': df['MACD_hist'].round(2).tolist() if 'MACD_hist' in df.columns else []
                },
                'macd_latest': round(latest_macd, 2) if latest_macd else None,
                'macd_trend': macd_signal,
                'bb_upper': df['BB_upper'].round(2).tolist() if 'BB_upper' in df.columns else [],
                'bb_middle': df['BB_middle'].round(2).tolist() if 'BB_middle' in df.columns else [],
                'bb_lower': df['BB_lower'].round(2).tolist() if 'BB_lower' in df.columns else [],
                'ma_20': df['MA_20'].round(2).tolist() if 'MA_20' in df.columns else [],
                'ma_50': df['MA_50'].round(2).tolist() if 'MA_50' in df.columns else [],
                'ma_100': df['MA_100'].round(2).tolist() if 'MA_100' in df.columns else [],
                'volume': df['Volume'].tolist(),
            }
            
            return result
        except Exception as e:
            print(f"Technical indicators error for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': f'Failed to calculate technical indicators for {symbol}. Please check the stock symbol and try again.'}
