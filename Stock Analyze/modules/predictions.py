import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb
import yfinance as yf
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        
    def add_technical_indicators(self, df):
        """Add comprehensive technical indicators to the dataframe"""
        try:
            # Moving Averages (different timeframes)
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_10'] = df['Close'].rolling(window=10).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            
            # Volume indicators
            df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # Price momentum and returns
            df['Daily_Return'] = df['Close'].pct_change()
            df['Return_5d'] = df['Close'].pct_change(periods=5)
            df['Return_10d'] = df['Close'].pct_change(periods=10)
            
            # Volatility indicators
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            
            # RSI (Relative Strength Index)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            # MACD
            macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            if macd is not None and not macd.empty:
                df['MACD'] = macd.iloc[:, 0]
                df['MACD_Signal'] = macd.iloc[:, 1]
                df['MACD_Hist'] = macd.iloc[:, 2]
            
            # Bollinger Bands
            bbands = ta.bbands(df['Close'], length=20, std=2)
            if bbands is not None and not bbands.empty:
                df['BB_Upper'] = bbands.iloc[:, 0]
                df['BB_Middle'] = bbands.iloc[:, 1]
                df['BB_Lower'] = bbands.iloc[:, 2]
                df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            
            # Stochastic Oscillator
            stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
            if stoch is not None and not stoch.empty:
                df['Stoch_K'] = stoch.iloc[:, 0]
                df['Stoch_D'] = stoch.iloc[:, 1]
            
            # Price position indicators
            df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
            df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']
            
            # Trend indicators
            df['Price_Trend'] = np.where(df['Close'] > df['MA_20'], 1, -1)
            df['MA_Cross'] = np.where(df['MA_5'] > df['MA_20'], 1, -1)
            
            return df
        except Exception as e:
            print(f"Error adding technical indicators: {e}")
            return df
        
    def prepare_data(self, symbol, period='2y'):
        """Prepare enhanced feature set with 2-3 years of data"""
        try:
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = f"{symbol}.NS"
            
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty or len(df) < 100:
                return None, None
            
            # Add all technical indicators
            df = self.add_technical_indicators(df)
            
            # Drop rows with NaN values
            df = df.dropna()
            
            if len(df) < 50:
                return None, None
            
            # Select features for training
            features = [
                'Open', 'High', 'Low', 'Volume',
                'MA_5', 'MA_10', 'MA_20', 'MA_50',
                'EMA_12', 'EMA_26',
                'Volume_MA', 'Volume_Ratio',
                'Daily_Return', 'Return_5d', 'Return_10d',
                'Volatility', 'ATR',
                'RSI',
                'High_Low_Pct', 'Close_Open_Pct',
                'Price_Trend', 'MA_Cross'
            ]
            
            # Add MACD features if available
            if 'MACD' in df.columns:
                features.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
            
            # Add Bollinger Bands if available
            if 'BB_Width' in df.columns:
                features.extend(['BB_Width'])
            
            # Add Stochastic if available  
            if 'Stoch_K' in df.columns:
                features.extend(['Stoch_K', 'Stoch_D'])
            
            # Filter to only features that exist
            available_features = [f for f in features if f in df.columns]
            
            X = df[available_features]
            y = df['Close']
            
            return X, y
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None, None
    
    def train_model(self, X, y, model_type='xgboost'):
        """Train model with feature scaling and optimized hyperparameters"""
        try:
            # Use walk-forward validation (time-series split)
            train_size = int(len(X) * 0.8)
            X_train, y_train = X[:train_size], y[:train_size]
            X_test, y_test = X[train_size:], y[train_size:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model based on type
            if model_type == 'linear_regression':
                model = LinearRegression()
                model.fit(X_train_scaled, y_train)
            elif model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
            else:  # xgboost (default)
                model = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_test_scaled, y_test)],
                    verbose=False
                )
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = np.mean(np.abs(y_test - y_pred))
            r2 = r2_score(y_test, y_pred) * 100
            
            # Directional accuracy (did we predict trend correctly?)
            actual_direction = np.sign(np.diff(y_test))
            predicted_direction = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            # Overall accuracy (inverse of MAPE)
            accuracy = max(0, 100 - mape)
            
            # Combined score (weighted average of metrics)
            combined_accuracy = (accuracy * 0.4 + directional_accuracy * 0.4 + r2 * 0.2) / 1.0
            
            metrics = {
                'mape': round(mape, 2),
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'r2': round(r2, 2),
                'accuracy': round(combined_accuracy, 2),
                'directional_accuracy': round(directional_accuracy, 2)
            }
            
            return model, metrics
        except Exception as e:
            print(f"Error training model: {e}")
            return None, None
    
    def predict(self, symbol, model_type='xgboost', prediction_range='1d'):
        """Make predictions using enhanced model"""
        try:
            X, y = self.prepare_data(symbol, period='2y')
            
            if X is None or y is None:
                return {'error': f'Unable to fetch historical data for {symbol}. Please ensure the stock symbol is correct and market data is available.'}
            
            model, metrics = self.train_model(X, y, model_type)
            
            if model is None:
                return {'error': f'Prediction model training failed. There may not be enough historical data for {symbol}.'}
            
            # Scale last data point
            last_data = X.iloc[-1:].values
            last_data_scaled = self.scaler.transform(last_data)
            
            days_map = {'1d': 1, '1w': 7, '1m': 30}
            days = days_map.get(prediction_range, 1)
            
            predictions = []
            current_data_scaled = last_data_scaled.copy()
            last_close = y.iloc[-1]
            
            # Generate predictions iteratively
            for i in range(days):
                pred = float(model.predict(current_data_scaled)[0])
                predictions.append(round(pred, 2))
                
                # Update features for next prediction
                if i < days - 1:
                    # Inverse transform to original space
                    current_data_raw = self.scaler.inverse_transform(current_data_scaled)[0]
                    
                    # Update raw features with predicted values
                    current_data_raw[0] = pred  # Update Open
                    current_data_raw[1] = pred * 1.005  # Update High
                    current_data_raw[2] = pred * 0.995  # Update Low
                    
                    # Re-scale the updated features
                    current_data_scaled = self.scaler.transform([current_data_raw])
            
            # Get historical data for comparison
            actual_prices = y.tail(60).round(2).tolist()
            actual_dates = y.tail(60).index.strftime('%Y-%m-%d').tolist()
            
            # Generate future dates
            future_dates = []
            last_date = y.index[-1]
            for i in range(1, days + 1):
                future_date = last_date + timedelta(days=i)
                future_dates.append(future_date.strftime('%Y-%m-%d'))
            
            # Calculate confidence based on metrics
            confidence = min(95, max(60, metrics['accuracy']))
            
            volatility = np.std(predictions) if len(predictions) > 1 else 0
            
            result = {
                'model_type': model_type.upper(),
                'predictions': predictions,
                'future_dates': future_dates,
                'actual_prices': actual_prices,
                'actual_dates': actual_dates,
                'metrics': metrics,
                'confidence': round(confidence, 2),
                'volatility': round(volatility, 2),
                'price_change': round(predictions[-1] - last_close, 2),
                'price_change_percent': round(((predictions[-1] - last_close) / last_close) * 100, 2),
            }
            
            return result
        except Exception as e:
            print(f"Prediction error: {e}")
            return {'error': f'Prediction failed for {symbol}. Please ensure the stock has sufficient historical data and try again.'}
    
    def get_ai_insights(self, symbol):
        """Generate AI insights with enhanced predictions"""
        try:
            prediction_data = self.predict(symbol, 'xgboost', '1w')
            
            if 'error' in prediction_data:
                return {'error': prediction_data['error']}
            
            price_change = prediction_data['price_change']
            price_change_percent = prediction_data['price_change_percent']
            confidence = prediction_data['confidence']
            
            # Determine sentiment
            if price_change_percent > 3:
                sentiment = 'Bullish'
                color = 'success'
                icon = '📈'
            elif price_change_percent < -3:
                sentiment = 'Bearish'
                color = 'danger'
                icon = '📉'
            else:
                sentiment = 'Neutral'
                color = 'warning'
                icon = '➡️'
            
            # Confidence level
            if confidence >= 80:
                confidence_level = 'High'
                confidence_color = 'success'
            elif confidence >= 65:
                confidence_level = 'Medium'
                confidence_color = 'info'
            else:
                confidence_level = 'Low'
                confidence_color = 'warning'
            
            direction = 'rise' if price_change > 0 else 'fall'
            summary = f"Based on advanced AI analysis using {prediction_data['metrics']['directional_accuracy']:.1f}% directional accuracy, "
            summary += f"{symbol} is expected to {direction} by ₹{abs(price_change):.2f} ({abs(price_change_percent):.2f}%) over the next week. "
            summary += f"Market sentiment is {sentiment.lower()} with {confidence_level.lower()} confidence ({confidence:.1f}%)."
            
            # Generate prediction text
            predicted_price = prediction_data['predictions'][-1] if prediction_data.get('predictions') else 0
            prediction_text = f"₹{predicted_price:.2f} (7-day forecast)"
            
            # Determine recommendation based on sentiment and confidence
            if sentiment == 'Bullish' and confidence >= 70:
                recommendation = 'Buy'
            elif sentiment == 'Bearish' and confidence >= 70:
                recommendation = 'Sell'
            else:
                recommendation = 'Hold'
            
            insights = {
                'sentiment': sentiment,
                'sentiment_color': color,
                'sentiment_icon': icon,
                'confidence': round(float(confidence), 2),
                'confidence_level': confidence_level,
                'confidence_color': confidence_color,
                'price_change': round(float(price_change), 2),
                'price_change_percent': round(float(price_change_percent), 2),
                'prediction': prediction_text,
                'recommendation': recommendation,
                'summary': summary,
                'accuracy': round(float(prediction_data['metrics']['accuracy']), 2),
                'directional_accuracy': round(float(prediction_data['metrics']['directional_accuracy']), 2)
            }
            
            return insights
        except Exception as e:
            print(f"Error generating insights: {e}")
            # Return fallback values instead of error
            return {
                'sentiment': 'Neutral',
                'sentiment_color': 'warning',
                'sentiment_icon': '➡️',
                'confidence': 50,
                'confidence_level': 'Low',
                'confidence_color': 'warning',
                'price_change': 0,
                'price_change_percent': 0,
                'prediction': 'Not enough data',
                'recommendation': 'Hold',
                'summary': f'Unable to generate AI insights for {symbol}. Insufficient historical data available.',
                'accuracy': 0,
                'directional_accuracy': 0
            }
