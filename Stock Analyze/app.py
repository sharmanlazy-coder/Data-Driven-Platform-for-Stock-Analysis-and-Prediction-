from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from modules.stock_data import StockDataFetcher
from modules.predictions import StockPredictor
from modules.capm import CAPMAnalyzer
from modules.technical_indicators import TechnicalAnalyzer
from modules.market_overview import MarketOverview
from modules.index_predictor import IndexPredictor
from models import db, User, Watchlist, DemoTrade
import json
import os
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stock_market.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

stock_fetcher = StockDataFetcher()
predictor = StockPredictor()
capm_analyzer = CAPMAnalyzer()
tech_analyzer = TechnicalAnalyzer()
market_overview = MarketOverview()
index_predictor = IndexPredictor()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('signup.html')
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            flash('Please enter a valid email address.', 'error')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('signup.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.', 'error')
            return render_template('signup.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please login instead.', 'error')
            return render_template('signup.html')
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'error')
            return render_template('login.html')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/stock/<symbol>')
@login_required
def stock_detail(symbol):
    return render_template('stock_detail.html', symbol=symbol)

@app.route('/index/<index_name>')
@login_required
def index_prediction(index_name):
    if index_name.lower() not in ['nifty50', 'sensex']:
        flash('Invalid index name.', 'error')
        return redirect(url_for('dashboard'))
    return render_template('index_prediction.html', index_name=index_name.lower())

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/api/search_stocks')
def search_stocks():
    query = request.args.get('query', '').upper()
    stocks = stock_fetcher.search_indian_stocks(query)
    return jsonify(stocks)

@app.route('/api/stock_data/<symbol>')
def get_stock_data(symbol):
    data = stock_fetcher.get_stock_data(symbol)
    return jsonify(data)

@app.route('/api/historical/<symbol>')
def get_historical_data(symbol):
    period = request.args.get('period', '6mo')
    data = stock_fetcher.get_historical_data(symbol, period)
    return jsonify(data)

@app.route('/api/predict/<symbol>')
def predict_stock(symbol):
    model_type = request.args.get('model', 'random_forest')
    prediction_range = request.args.get('range', '1d')
    
    prediction_data = predictor.predict(symbol, model_type, prediction_range)
    return jsonify(prediction_data)

@app.route('/api/capm/<symbol>')
def get_capm_analysis(symbol):
    capm_data = capm_analyzer.analyze(symbol)
    return jsonify(capm_data)

@app.route('/api/technical/<symbol>')
def get_technical_indicators(symbol):
    indicators = tech_analyzer.get_all_indicators(symbol)
    return jsonify(indicators)

@app.route('/api/market_overview')
def get_market_overview():
    overview = market_overview.get_overview()
    return jsonify(overview)

@app.route('/api/top_movers')
def get_top_movers():
    movers = market_overview.get_top_movers()
    return jsonify(movers)

@app.route('/api/cap_wise_movers')
def get_cap_wise_movers():
    cap_type = request.args.get('cap', 'large')
    movers = market_overview.get_cap_wise_movers(cap_type)
    return jsonify(movers)

@app.route('/api/ai_insights/<symbol>')
def get_ai_insights(symbol):
    insights = predictor.get_ai_insights(symbol)
    return jsonify(insights)

@app.route('/api/sector_performance')
def get_sector_performance():
    sectors = market_overview.get_sector_performance()
    return jsonify(sectors)

@app.route('/api/index_data/<index_name>')
def get_index_data(index_name):
    period = request.args.get('period', '3mo')
    data = index_predictor.get_index_data(index_name, period)
    return jsonify(data)

@app.route('/api/index_prediction/<index_name>')
def get_index_prediction(index_name):
    days = int(request.args.get('days', 7))
    prediction = index_predictor.predict_index(index_name, days)
    return jsonify(prediction)

@app.route('/profile')
@login_required
def profile():
    total_trades = DemoTrade.query.filter_by(user_id=current_user.id).count()
    open_trades = DemoTrade.query.filter_by(user_id=current_user.id, status='OPEN').count()
    closed_trades = DemoTrade.query.filter_by(user_id=current_user.id, status='CLOSED').all()
    
    total_profit_loss = sum(trade.profit_loss for trade in closed_trades if trade.profit_loss)
    
    return render_template('profile.html', 
                          total_trades=total_trades,
                          open_trades=open_trades,
                          total_profit_loss=total_profit_loss)

@app.route('/watchlist')
@login_required
def view_watchlist():
    user_watchlist = Watchlist.query.filter_by(user_id=current_user.id).all()
    watchlist_data = []
    for item in user_watchlist:
        stock_data = stock_fetcher.get_stock_data(item.stock_symbol)
        if 'error' not in stock_data:
            watchlist_data.append({
                'symbol': item.stock_symbol,
                'name': stock_data.get('name', item.stock_symbol),
                'price': stock_data.get('current_price', 0),
                'change': stock_data.get('change', 0),
                'change_percent': stock_data.get('change_percent', 0)
            })
    return render_template('watchlist.html', watchlist=watchlist_data)

@app.route('/api/watchlist', methods=['GET', 'POST', 'DELETE'])
@login_required
def manage_watchlist():
    if request.method == 'GET':
        user_watchlist = Watchlist.query.filter_by(user_id=current_user.id).all()
        return jsonify({'symbols': [item.stock_symbol for item in user_watchlist]})
    elif request.method == 'POST':
        if not request.json or 'symbol' not in request.json:
            return jsonify({'error': 'Please provide a stock symbol in JSON format: {"symbol": "STOCK.NS"}'}), 400
        symbol = request.json.get('symbol')
        if not symbol:
            return jsonify({'error': 'Stock symbol cannot be empty'}), 400
        
        existing = Watchlist.query.filter_by(user_id=current_user.id, stock_symbol=symbol).first()
        if existing:
            return jsonify({'message': 'Stock already in watchlist', 'symbol': symbol})
        
        new_watchlist_item = Watchlist(user_id=current_user.id, stock_symbol=symbol)
        db.session.add(new_watchlist_item)
        db.session.commit()
        return jsonify({'message': 'Stock added to watchlist', 'symbol': symbol})
    elif request.method == 'DELETE':
        if not request.json or 'symbol' not in request.json:
            return jsonify({'error': 'Please provide a stock symbol in JSON format: {"symbol": "STOCK.NS"}'}), 400
        symbol = request.json.get('symbol')
        if not symbol:
            return jsonify({'error': 'Stock symbol cannot be empty'}), 400
        
        item = Watchlist.query.filter_by(user_id=current_user.id, stock_symbol=symbol).first()
        if item:
            db.session.delete(item)
            db.session.commit()
            return jsonify({'message': 'Stock removed from watchlist', 'symbol': symbol})
        return jsonify({'error': 'Stock not found in watchlist'}), 404

@app.route('/demo-trading')
@login_required
def demo_trading():
    trades = DemoTrade.query.filter_by(user_id=current_user.id).order_by(DemoTrade.created_at.desc()).all()
    open_positions = [t for t in trades if t.status == 'OPEN']
    closed_trades = [t for t in trades if t.status == 'CLOSED']
    
    portfolio = {}
    for trade in open_positions:
        if trade.symbol not in portfolio:
            portfolio[trade.symbol] = {'quantity': 0, 'total_invested': 0, 'trades': []}
        if trade.trade_type == 'BUY':
            portfolio[trade.symbol]['quantity'] += trade.quantity
            portfolio[trade.symbol]['total_invested'] += trade.total_value
        else:
            portfolio[trade.symbol]['quantity'] -= trade.quantity
            portfolio[trade.symbol]['total_invested'] -= trade.total_value
        portfolio[trade.symbol]['trades'].append(trade)
    
    for symbol in portfolio:
        if portfolio[symbol]['quantity'] > 0:
            current_data = stock_fetcher.get_stock_data(symbol)
            if 'error' not in current_data:
                current_price = current_data.get('current_price', 0)
                current_value = current_price * portfolio[symbol]['quantity']
                portfolio[symbol]['current_price'] = current_price
                portfolio[symbol]['current_value'] = current_value
                portfolio[symbol]['profit_loss'] = current_value - portfolio[symbol]['total_invested']
                portfolio[symbol]['profit_loss_percent'] = ((current_value - portfolio[symbol]['total_invested']) / portfolio[symbol]['total_invested'] * 100) if portfolio[symbol]['total_invested'] > 0 else 0
    
    return render_template('demo_trading.html', 
                         balance=current_user.demo_balance, 
                         portfolio=portfolio,
                         open_trades=open_positions,
                         closed_trades=closed_trades)

@app.route('/api/demo-trade', methods=['POST'])
@login_required
def execute_demo_trade():
    if not request.json:
        return jsonify({'error': 'Invalid request format'}), 400
    
    symbol = request.json.get('symbol')
    trade_type = request.json.get('trade_type')
    quantity = request.json.get('quantity')
    stoploss = request.json.get('stoploss')
    target = request.json.get('target')
    
    if not symbol or not trade_type or not quantity:
        return jsonify({'error': 'Symbol, trade type, and quantity are required'}), 400
    
    if trade_type not in ['BUY', 'SELL']:
        return jsonify({'error': 'Trade type must be BUY or SELL'}), 400
    
    try:
        quantity = int(quantity)
        if quantity <= 0:
            return jsonify({'error': 'Quantity must be positive'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid quantity'}), 400
    
    stock_data = stock_fetcher.get_stock_data(symbol)
    if 'error' in stock_data:
        return jsonify({'error': 'Unable to fetch stock price'}), 400
    
    price = stock_data.get('current_price', 0)
    total_value = price * quantity
    
    if trade_type == 'BUY':
        if current_user.demo_balance < total_value:
            return jsonify({'error': f'Insufficient balance. Required: ₹{total_value:.2f}, Available: ₹{current_user.demo_balance:.2f}'}), 400
        current_user.demo_balance -= total_value
    else:
        current_user.demo_balance += total_value
    
    new_trade = DemoTrade(
        user_id=current_user.id,
        symbol=symbol,
        trade_type=trade_type,
        price=price,
        quantity=quantity,
        total_value=total_value,
        stoploss=float(stoploss) if stoploss else None,
        target=float(target) if target else None,
        status='OPEN'
    )
    
    db.session.add(new_trade)
    db.session.commit()
    
    return jsonify({
        'message': f'{trade_type} order executed successfully',
        'trade_id': new_trade.id,
        'symbol': symbol,
        'quantity': quantity,
        'price': price,
        'total_value': total_value,
        'balance': current_user.demo_balance
    })

@app.route('/api/close-trade/<int:trade_id>', methods=['POST'])
@login_required
def close_trade(trade_id):
    trade = DemoTrade.query.filter_by(id=trade_id, user_id=current_user.id).first()
    if not trade:
        return jsonify({'error': 'Trade not found'}), 404
    
    if trade.status != 'OPEN':
        return jsonify({'error': 'Trade is already closed'}), 400
    
    stock_data = stock_fetcher.get_stock_data(trade.symbol)
    if 'error' in stock_data:
        return jsonify({'error': 'Unable to fetch current stock price'}), 400
    
    current_price = stock_data.get('current_price', 0)
    current_value = current_price * trade.quantity
    
    if trade.trade_type == 'BUY':
        profit_loss = current_value - trade.total_value
        current_user.demo_balance += current_value
    else:
        profit_loss = trade.total_value - current_value
        current_user.demo_balance -= current_value
    
    trade.status = 'CLOSED'
    trade.closed_at = db.func.now()
    trade.profit_loss = profit_loss
    
    db.session.commit()
    
    return jsonify({
        'message': 'Trade closed successfully',
        'profit_loss': profit_loss,
        'balance': current_user.demo_balance
    })

@app.route('/ipo')
def ipo_listing():
    return render_template('ipo.html')

@app.route('/compare')
def stock_comparison():
    return render_template('compare.html')

@app.route('/api/compare-stocks', methods=['GET', 'POST'])
def compare_stocks_api():
    try:
        if request.method == 'POST':
            if not request.json or 'symbols' not in request.json:
                return jsonify({'error': 'Please provide stock symbols in JSON format: {"symbols": ["STOCK1", "STOCK2", ...]}'}), 400
            
            symbols = request.json['symbols']
            if not isinstance(symbols, list):
                return jsonify({'error': 'Symbols must be provided as a list'}), 400
            
            if len(symbols) < 2:
                return jsonify({'error': 'Please provide at least 2 symbols to compare'}), 400
            
            if len(symbols) > 5:
                return jsonify({'error': 'Maximum 5 stocks can be compared at once'}), 400
        else:
            # Handle legacy GET request with symbol1 and symbol2 parameters
            symbol1 = request.args.get('symbol1', '').upper()
            symbol2 = request.args.get('symbol2', '').upper()
            
            if not symbol1 or not symbol2:
                return jsonify({'error': 'Both symbols are required'}), 400
            
            symbols = [symbol1, symbol2]

        # Process all symbols
        stock_data = {}
        for symbol in symbols:
            # Add .NS suffix if not present
            symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
            data = stock_fetcher.get_stock_data(symbol)
            
            if 'error' in data:
                return jsonify({'error': f'Unable to fetch data for {symbol}'}), 400
            
            # Standardize the data structure
            if 'current_price' not in data:
                data['current_price'] = None
            if 'change' not in data:
                data['change'] = None
            if 'change_percent' not in data:
                data['change_percent'] = None
            if 'high' not in data:
                data['high'] = data.get('day_high')
            if 'low' not in data:
                data['low'] = data.get('day_low')
            if '52_week_high' not in data:
                data['52_week_high'] = data.get('week_52_high')
            if '52_week_low' not in data:
                data['52_week_low'] = data.get('week_52_low')
            
            stock_data[symbol] = data
        
        return jsonify({'stocks': stock_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
