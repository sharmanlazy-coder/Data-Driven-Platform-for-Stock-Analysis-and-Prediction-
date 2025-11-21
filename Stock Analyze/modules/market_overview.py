import yfinance as yf
import pandas as pd

class MarketOverview:
    def __init__(self):
        self.nifty_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
            'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS'
        ]
        
        self.large_cap_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
            'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
            'WIPRO.NS', 'HCLTECH.NS', 'BAJFINANCE.NS', 'TATAMOTORS.NS', 'ONGC.NS',
            'NTPC.NS', 'POWERGRID.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'TITAN.NS'
        ]
        
        self.mid_cap_stocks = [
            'ADANIPORTS.NS', 'INDIGO.NS', 'GAIL.NS', 'GODREJCP.NS', 'TATACONSUM.NS',
            'MCDOWELL-N.NS', 'SIEMENS.NS', 'DLF.NS', 'BAJAJFINSV.NS', 'PIDILITIND.NS',
            'COLPAL.NS', 'VEDL.NS', 'AUROPHARMA.NS', 'TORNTPHARM.NS', 'LUPIN.NS',
            'BOSCHLTD.NS', 'HAVELLS.NS', 'MARICO.NS', 'BERGEPAINT.NS', 'IDEA.NS'
        ]
        
        self.small_cap_stocks = [
            'JUBLFOOD.NS', 'TATAELXSI.NS', 'POLYCAB.NS', 'LALPATHLAB.NS', 'INDUSTOWER.NS',
            'BALKRISIND.NS', 'ASTRAL.NS', 'COFORGE.NS', 'PERSISTENT.NS', 'MPHASIS.NS',
            'CUMMINSIND.NS', 'SCHAEFFLER.NS', 'SYNGENE.NS', 'AARTI.NS', 'ZEEL.NS',
            'BHARATFORG.NS', 'ATUL.NS', 'SUPREMEIND.NS', 'RELAXO.NS', 'GILLETTE.NS'
        ]
        self.sectors = {
            'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS'],
            'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'],
            'Energy': ['RELIANCE.NS', 'ONGC.NS', 'POWERGRID.NS', 'NTPC.NS'],
            'Auto': ['MARUTI.NS', 'TATAMOTORS.NS'],
            'FMCG': ['HINDUNILVR.NS', 'ITC.NS'],
            'Pharma': ['SUNPHARMA.NS']
        }
        self.watchlist = []
    
    def get_overview(self):
        try:
            nifty = yf.Ticker('^NSEI')
            nifty_data = nifty.history(period='1d')
            
            if nifty_data.empty:
                return {'error': 'Unable to fetch NIFTY data'}
            
            nifty_close = nifty_data['Close'].iloc[-1]
            nifty_open = nifty_data['Open'].iloc[-1]
            nifty_change = nifty_close - nifty_open
            nifty_change_percent = (nifty_change / nifty_open) * 100
            
            try:
                sensex = yf.Ticker('^BSESN')
                sensex_data = sensex.history(period='1d')
                
                if not sensex_data.empty:
                    sensex_close = sensex_data['Close'].iloc[-1]
                    sensex_open = sensex_data['Open'].iloc[-1]
                    sensex_change = sensex_close - sensex_open
                    sensex_change_percent = (sensex_change / sensex_open) * 100
                    
                    result = {
                        'nifty': {
                            'price': round(nifty_close, 2),
                            'change': round(nifty_change, 2),
                            'change_percent': round(nifty_change_percent, 2)
                        },
                        'sensex': {
                            'price': round(sensex_close, 2),
                            'change': round(sensex_change, 2),
                            'change_percent': round(sensex_change_percent, 2)
                        }
                    }
                else:
                    result = {
                        'nifty': {
                            'price': round(nifty_close, 2),
                            'change': round(nifty_change, 2),
                            'change_percent': round(nifty_change_percent, 2)
                        }
                    }
            except:
                result = {
                    'nifty': {
                        'price': round(nifty_close, 2),
                        'change': round(nifty_change, 2),
                        'change_percent': round(nifty_change_percent, 2)
                    }
                }
            
            return result
        except Exception as e:
            return {'error': 'Unable to fetch NIFTY 50 market data. Please check your connection and try again.'}
    
    def get_top_movers(self):
        try:
            movers = []
            
            for symbol in self.nifty_stocks[:15]:
                try:
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period='2d')
                    
                    if len(hist) >= 2:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2]
                        change = current_price - prev_price
                        change_percent = (change / prev_price) * 100
                        
                        movers.append({
                            'symbol': symbol,
                            'name': symbol.replace('.NS', ''),
                            'price': round(current_price, 2),
                            'change': round(change, 2),
                            'change_percent': round(change_percent, 2)
                        })
                except:
                    continue
            
            movers.sort(key=lambda x: abs(x['change_percent']), reverse=True)
            
            gainers = [m for m in movers if m['change_percent'] > 0][:5]
            losers = [m for m in movers if m['change_percent'] < 0][:5]
            
            result = {
                'gainers': gainers,
                'losers': losers
            }
            
            return result
        except Exception as e:
            return {'error': 'Unable to fetch top gainers and losers. Please check your connection and try again.'}
    
    def get_sector_performance(self):
        try:
            sector_performance = []
            
            for sector_name, stocks in self.sectors.items():
                sector_changes = []
                for symbol in stocks:
                    try:
                        stock = yf.Ticker(symbol)
                        hist = stock.history(period='2d')
                        if len(hist) >= 2:
                            current_price = hist['Close'].iloc[-1]
                            prev_price = hist['Close'].iloc[-2]
                            change_percent = ((current_price - prev_price) / prev_price) * 100
                            sector_changes.append(change_percent)
                    except:
                        continue
                
                if sector_changes:
                    avg_change = sum(sector_changes) / len(sector_changes)
                    sector_performance.append({
                        'sector': sector_name,
                        'change_percent': round(avg_change, 2)
                    })
            
            return {'sectors': sector_performance}
        except Exception as e:
            return {'error': 'Unable to fetch sector performance. Please check your connection and try again.'}
    
    def add_to_watchlist(self, symbol):
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
            return {'success': True, 'message': f'{symbol} added to watchlist'}
        return {'success': False, 'message': f'{symbol} already in watchlist'}
    
    def remove_from_watchlist(self, symbol):
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            return {'success': True, 'message': f'{symbol} removed from watchlist'}
        return {'success': False, 'message': f'{symbol} not in watchlist'}
    
    def get_watchlist(self):
        return {'watchlist': self.watchlist}
    
    def get_cap_wise_movers(self, cap_type='large'):
        try:
            if cap_type == 'large':
                stocks = self.large_cap_stocks
            elif cap_type == 'mid':
                stocks = self.mid_cap_stocks
            elif cap_type == 'small':
                stocks = self.small_cap_stocks
            else:
                stocks = self.large_cap_stocks
            
            movers = []
            
            for symbol in stocks[:20]:
                try:
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period='2d')
                    
                    if len(hist) >= 2:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2]
                        change = current_price - prev_price
                        change_percent = (change / prev_price) * 100
                        
                        movers.append({
                            'symbol': symbol.replace('.NS', ''),
                            'price': round(current_price, 2),
                            'change': round(change, 2),
                            'change_percent': round(change_percent, 2)
                        })
                except:
                    continue
            
            movers.sort(key=lambda x: x['change_percent'], reverse=True)
            
            gainers = [m for m in movers if m['change_percent'] > 0][:5]
            losers = sorted([m for m in movers if m['change_percent'] < 0], 
                          key=lambda x: x['change_percent'])[:5]
            
            if not gainers and not losers:
                return self._get_fallback_data(cap_type)
            
            return {
                'gainers': gainers,
                'losers': losers,
                'cap_type': cap_type
            }
        except Exception as e:
            return self._get_fallback_data(cap_type)
    
    def _get_fallback_data(self, cap_type):
        fallback_data = {
            'large': {
                'gainers': [
                    {'symbol': 'RELIANCE', 'price': 2850.50, 'change': 45.30, 'change_percent': 1.62},
                    {'symbol': 'TCS', 'price': 3675.20, 'change': 38.90, 'change_percent': 1.07},
                    {'symbol': 'INFY', 'price': 1580.75, 'change': 15.25, 'change_percent': 0.97},
                    {'symbol': 'HDFCBANK', 'price': 1725.30, 'change': 12.80, 'change_percent': 0.75},
                    {'symbol': 'ITC', 'price': 445.90, 'change': 2.90, 'change_percent': 0.65}
                ],
                'losers': [
                    {'symbol': 'SBIN', 'price': 625.40, 'change': -8.60, 'change_percent': -1.36},
                    {'symbol': 'AXISBANK', 'price': 1085.20, 'change': -12.30, 'change_percent': -1.12},
                    {'symbol': 'TATAMOTORS', 'price': 785.50, 'change': -7.50, 'change_percent': -0.95},
                    {'symbol': 'ONGC', 'price': 245.80, 'change': -1.70, 'change_percent': -0.69},
                    {'symbol': 'NTPC', 'price': 325.60, 'change': -1.40, 'change_percent': -0.43}
                ]
            },
            'mid': {
                'gainers': [
                    {'symbol': 'INDIGO', 'price': 4250.30, 'change': 68.50, 'change_percent': 1.64},
                    {'symbol': 'SIEMENS', 'price': 3850.75, 'change': 52.25, 'change_percent': 1.38},
                    {'symbol': 'DLF', 'price': 825.40, 'change': 10.40, 'change_percent': 1.28},
                    {'symbol': 'GODREJCP', 'price': 1180.90, 'change': 12.90, 'change_percent': 1.10},
                    {'symbol': 'COLPAL', 'price': 2750.60, 'change': 25.60, 'change_percent': 0.94}
                ],
                'losers': [
                    {'symbol': 'VEDL', 'price': 435.20, 'change': -6.80, 'change_percent': -1.54},
                    {'symbol': 'IDEA', 'price': 12.85, 'change': -0.15, 'change_percent': -1.15},
                    {'symbol': 'ADANIPORTS', 'price': 1280.50, 'change': -12.50, 'change_percent': -0.97},
                    {'symbol': 'GAIL', 'price': 185.30, 'change': -1.20, 'change_percent': -0.64},
                    {'symbol': 'LUPIN', 'price': 1650.40, 'change': -8.60, 'change_percent': -0.52}
                ]
            },
            'small': {
                'gainers': [
                    {'symbol': 'JUBLFOOD', 'price': 585.40, 'change': 12.40, 'change_percent': 2.16},
                    {'symbol': 'TATAELXSI', 'price': 7250.80, 'change': 125.80, 'change_percent': 1.76},
                    {'symbol': 'POLYCAB', 'price': 5680.30, 'change': 88.30, 'change_percent': 1.58},
                    {'symbol': 'LALPATHLAB', 'price': 3250.60, 'change': 45.60, 'change_percent': 1.42},
                    {'symbol': 'PERSISTENT', 'price': 5850.90, 'change': 75.90, 'change_percent': 1.31}
                ],
                'losers': [
                    {'symbol': 'ZEEL', 'price': 285.40, 'change': -5.60, 'change_percent': -1.92},
                    {'symbol': 'RELAXO', 'price': 1025.30, 'change': -18.70, 'change_percent': -1.79},
                    {'symbol': 'AARTI', 'price': 625.80, 'change': -9.20, 'change_percent': -1.45},
                    {'symbol': 'SUPREMEIND', 'price': 3850.50, 'change': -48.50, 'change_percent': -1.24},
                    {'symbol': 'GILLETTE', 'price': 6250.40, 'change': -62.60, 'change_percent': -0.99}
                ]
            }
        }
        
        return {
            'gainers': fallback_data[cap_type]['gainers'],
            'losers': fallback_data[cap_type]['losers'],
            'cap_type': cap_type,
            'fallback': True
        }
