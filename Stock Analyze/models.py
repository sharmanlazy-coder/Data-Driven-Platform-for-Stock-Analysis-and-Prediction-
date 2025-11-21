from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    demo_balance = db.Column(db.Float, default=100000.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    watchlist = db.relationship('Watchlist', backref='user', lazy=True, cascade='all, delete-orphan')
    trades = db.relationship('DemoTrade', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'

class Watchlist(db.Model):
    __tablename__ = 'watchlist'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    stock_symbol = db.Column(db.String(20), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('user_id', 'stock_symbol', name='unique_user_stock'),)
    
    def __repr__(self):
        return f'<Watchlist {self.stock_symbol} for User {self.user_id}>'

class DemoTrade(db.Model):
    __tablename__ = 'demo_trades'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    trade_type = db.Column(db.String(10), nullable=False)
    price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    total_value = db.Column(db.Float, nullable=False)
    stoploss = db.Column(db.Float, nullable=True)
    target = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), default='OPEN')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    closed_at = db.Column(db.DateTime, nullable=True)
    profit_loss = db.Column(db.Float, default=0.0)
    
    __table_args__ = (
        db.CheckConstraint("trade_type IN ('BUY', 'SELL')", name='check_trade_type'),
        db.CheckConstraint("status IN ('OPEN', 'CLOSED', 'CANCELLED')", name='check_status'),
    )
    
    def __repr__(self):
        return f'<DemoTrade {self.trade_type} {self.quantity} {self.symbol} @ ₹{self.price}>'
