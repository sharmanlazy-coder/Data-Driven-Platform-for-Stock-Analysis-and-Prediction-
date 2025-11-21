import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class CAPMAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.07
        
    def calculate_beta(self, stock_returns, market_returns):
        try:
            covariance = np.cov(stock_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance != 0 else 1
            return beta
        except:
            return 1
    
    def analyze(self, symbol):
        try:
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = f"{symbol}.NS"
            
            stock = yf.Ticker(symbol)
            nifty = yf.Ticker('^NSEI')
            
            stock_hist = stock.history(period='1y')
            nifty_hist = nifty.history(period='1y')
            
            if stock_hist.empty or nifty_hist.empty:
                return {'error': f'Unable to fetch historical data for CAPM analysis. Please ensure {symbol} is a valid Indian stock symbol.'}
            
            stock_returns = stock_hist['Close'].pct_change().dropna()
            nifty_returns = nifty_hist['Close'].pct_change().dropna()
            
            common_dates = stock_returns.index.intersection(nifty_returns.index)
            stock_returns = stock_returns.loc[common_dates]
            nifty_returns = nifty_returns.loc[common_dates]
            
            beta = self.calculate_beta(stock_returns.values, nifty_returns.values)
            
            market_return = nifty_returns.mean() * 252
            
            expected_return = self.risk_free_rate + beta * (market_return - self.risk_free_rate)
            
            actual_return = stock_returns.mean() * 252
            alpha = actual_return - expected_return
            
            sml_data = []
            beta_range = np.linspace(0, 2, 50)
            for b in beta_range:
                er = self.risk_free_rate + b * (market_return - self.risk_free_rate)
                sml_data.append({
                    'beta': round(b, 2),
                    'expected_return': round(er * 100, 2)
                })
            
            scatter_data = {
                'stock_return': round(actual_return * 100, 2),
                'market_return': round(market_return * 100, 2),
                'stock_risk': round(stock_returns.std() * np.sqrt(252) * 100, 2),
                'market_risk': round(nifty_returns.std() * np.sqrt(252) * 100, 2)
            }
            
            result = {
                'beta': round(beta, 3),
                'expected_return': round(expected_return * 100, 2),
                'actual_return': round(actual_return * 100, 2),
                'alpha': round(alpha * 100, 2),
                'risk_free_rate': round(self.risk_free_rate * 100, 2),
                'market_return': round(market_return * 100, 2),
                'excess_return': round((market_return - self.risk_free_rate) * 100, 2),
                'sml_data': sml_data,
                'scatter_data': scatter_data,
                'interpretation': self.get_interpretation(beta, alpha)
            }
            
            return result
        except Exception as e:
            return {'error': f'CAPM analysis failed for {symbol}. Please ensure the stock symbol is valid and try again.'}
    
    def get_interpretation(self, beta, alpha):
        interpretations = []
        
        if beta > 1.2:
            interpretations.append(f"High beta ({beta:.2f}) indicates high volatility - stock moves {beta:.1f}x market movement")
        elif beta < 0.8:
            interpretations.append(f"Low beta ({beta:.2f}) indicates low volatility - defensive stock")
        else:
            interpretations.append(f"Moderate beta ({beta:.2f}) - stock moves with the market")
        
        if alpha > 0.02:
            interpretations.append(f"Positive alpha ({alpha*100:.2f}%) - outperforming expected returns")
        elif alpha < -0.02:
            interpretations.append(f"Negative alpha ({alpha*100:.2f}%) - underperforming expected returns")
        else:
            interpretations.append("Alpha near zero - performing as expected")
        
        return ' | '.join(interpretations)
