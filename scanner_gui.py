import tkinter as tk
from tkinter import ttk
import logging
from market_scanner import MarketScanner
import threading
import pandas as pd
import mplfinance as mpf
from datetime import datetime
import json
import sqlite3
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import ccxt
import talib
import os

class ScannerGUI(MarketScanner):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Market Scanner")
        self.master.geometry("1200x800")
        self.master.configure(bg='#1E1E1E')
        
        # Initialize core components
        self.setup_logger()
        self.setup_database()
        self.setup_variables()
        self.setup_exchange()
        
        # Setup GUI components
        self.setup_styles()
        self.create_main_layout()
        self.create_navigation_bar()
        self.create_left_sidebar()
        self.create_center_panel()
        self.create_right_panel()
        self.create_bottom_panel()
        
        # Try auto-login
        if not self.auto_login():
            self.create_auth_window()
        else:
            self.start_scanning()

    def setup_styles(self):
        style = ttk.Style()
        style.configure('Dashboard.TFrame', background='#1E1E1E')
        style.configure('Dashboard.TLabel', background='#1E1E1E', foreground='white')
        style.configure('Dashboard.TButton', background='#2C2C2C', foreground='white')
        style.configure('Auth.TLabel', background='#1E1E1E', foreground='white', font=('Helvetica', 10))
        style.configure('Error.TLabel', background='#1E1E1E', foreground='red', font=('Helvetica', 10))

    def on_closing(self):
        """Handle window closing properly"""
        if self.scanning:
            self.stop_scanning()
        self.cleanup()
        self.master.quit()
        self.master.destroy()

    def cleanup(self):
        """Enhanced cleanup with proper event handling"""
        self.scanning = False
        
        if hasattr(self, 'active_streams'):
            self.active_streams.clear()
            
        if hasattr(self, 'binance'):
            try:
                if isinstance(self.binance, ccxt.binance):
                    self.binance = None
                self.logger.info("Exchange connection closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing exchange connection: {e}")
                
        if hasattr(self, 'save_favorites'):
            self.save_favorites()

        
        # Try auto-login first
        if not self.auto_login():
            # Initialize exchange connection
            self.initialize_exchange()
            # Setup GUI after exchange initialization
            self.setup_gui()
            # Show authentication window only if auto-login fails
            self.create_auth_window()
        else:
            # Setup GUI after successful auto-login
            self.setup_gui()
            self.start_scanning()

    def setup_logger(self, log_file='logs/scanner.log'):
        os.makedirs('logs', exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # File handler with rotation
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)
        
    def setup_database(self):
        self.db_connect = "connection.db"
        self.db_signals = "Signals.db"
        self.sql_operations('create', self.db_connect, 'userinfo')
        self.sql_operations('create', self.db_signals, 'Signals')
        
    def setup_variables(self):
        self.scanning = False
        self.current_market = None
        self.timeframe_var = tk.StringVar(value='15m')
        self.exchange_var = tk.StringVar(value='Binance')
        self.filter_var = tk.StringVar(value='Best-vol')
        self.user_choice = tk.IntVar(value=0)
        self.active_streams = set()
        self.choose_list = tk.StringVar(value='USDT')
        self.choose_time = tk.StringVar(value='15m')
        self.binance = None
        
        # Authentication variables
        self.api_key = tk.StringVar()
        self.api_secret = tk.StringVar()
        self.is_authenticated = False
    def setup_exchange(self):
        """Initialize exchange connection with default settings"""
        try:
            self.binance = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'recvWindow': 5000
                },
                'timeout': 30000
            })
            
            # Load markets
            if self.is_authenticated:
                self.binance.load_markets()
                self.logger.info("Exchange connection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Exchange setup error: {e}")
    def auto_login(self):
        saved_data = self.sql_operations('fetch', self.db_connect, 'userinfo')
        if saved_data:
            last_login = saved_data[-1]
            try:
                # Clear any existing invalid credentials
                self.sql_operations('delete', self.db_connect, 'userinfo')
                
                # Initialize exchange with saved credentials
                self.binance = ccxt.binance({
                    'apiKey': last_login['key'].strip(),
                    'secret': last_login['secret'].strip(),
                    'enableRateLimit': True,
                    'options': {
                        'adjustForTimeDifference': True,
                        'recvWindow': 5000
                    }
                })
                
                # Verify API access
                self.binance.fetch_balance()
                self.binance.load_markets()
                
                # Credentials valid - restore settings
                self.tel_id = last_login['tel_id']
                self.bot_token = last_login['tel_token']
                self.is_authenticated = True
                
                # Initialize scanner
                self.setup_market_streams()
                self.master.title(f"Market Scanner - {last_login['name']}")
                self.start_scanning()
                
                self.logger.info(f"Login successful for user: {last_login['name']}")
                return True
                
            except Exception as e:
                self.logger.info("Please enter valid API credentials")
                return False
                
        return False


    def show_error(self, title, message):
        """Display error message in a popup window"""
        error_window = tk.Toplevel(self.master)
        error_window.title(title)
        error_window.geometry("300x150")
        error_window.configure(bg='#1E1E1E')
        
        # Center the window
        error_window.geometry("+%d+%d" % (
            self.master.winfo_rootx() + 50,
            self.master.winfo_rooty() + 50))
        
        # Error message
        ttk.Label(error_window, 
                text=message,
                style='Error.TLabel',
                wraplength=250).pack(pady=20)
        
        # OK button
        ttk.Button(error_window,
                text="OK",
                command=error_window.destroy).pack(pady=10)
        
        # Make window modal
        error_window.transient(self.master)
        error_window.grab_set()
        self.master.wait_window(error_window)
    def create_auth_window(self):
        self.auth_window = tk.Toplevel(self.master)
        self.auth_window.title("Login to Market Scanner")
        self.auth_window.geometry("400x500")
        self.auth_window.configure(bg='#1E1E1E')
        
        # Fetch stored credentials immediately
        stored_credentials = self.sql_operations('fetch', self.db_connect, 'userinfo')
        last_login = stored_credentials[-1] if stored_credentials else {}
        
        # Create frames
        header_frame = ttk.Frame(self.auth_window, style='Auth.TFrame')
        header_frame.pack(fill='x', padx=20, pady=20)
        
        form_frame = ttk.Frame(self.auth_window, style='Auth.TFrame')
        form_frame.pack(fill='both', expand=True, padx=20)
        
        # Header
        ttk.Label(header_frame, text="Market Scanner Login",
                font=('Helvetica', 16, 'bold'),
                style='Auth.TLabel').pack()
        
        # Form fields with auto-population
        fields = [
            ('Name:', 'name', False),
            ('API Key:', 'key', True),
            ('API Secret:', 'secret', True),
            ('Passphrase:', 'phrase', True),
            ('Telegram ID:', 'tel_id', False),
            ('Telegram Token:', 'tel_token', True)
        ]
        
        self.auth_vars = {}
        for label, key, is_secure in fields:
            field_frame = ttk.Frame(form_frame)
            field_frame.pack(fill='x', pady=10)
            
            ttk.Label(field_frame, text=label, style='Auth.TLabel').pack(anchor='w')
            
            # Set stored value from database
            stored_value = last_login.get(key, '')
            self.auth_vars[key] = tk.StringVar(value=stored_value)
            
            # Create entry with stored value
            entry = ttk.Entry(field_frame, textvariable=self.auth_vars[key])
            if is_secure:
                entry.configure(show='*')
            entry.pack(fill='x')
            
            # Insert stored value
            if stored_value:
                entry.delete(0, tk.END)
                entry.insert(0, stored_value)
        
        # Buttons
        button_frame = ttk.Frame(form_frame)
        button_frame.pack(fill='x', pady=20)
        
        ttk.Button(button_frame, text="Login",
                command=self.process_login).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Clear Fields",
                command=lambda: self.clear_fields()).pack(side='right', padx=5)

def clear_fields(self):
    """Clear all authentication fields"""
    for var in self.auth_vars.values():
        var.set('')

    def process_login(self):
        auth_data = {key: var.get() for key, var in self.auth_vars.items()}
        
        if all(auth_data.values()):
            try:
                # Test API connection first
                test_exchange = ccxt.binance({
                    'apiKey': auth_data['key'].strip(),
                    'secret': auth_data['secret'].strip(),
                    'enableRateLimit': True
                })
                
                # Verify API keys by making a test request
                test_exchange.fetch_balance()
                
                # If successful, store credentials and initialize main exchange
                self.binance = test_exchange
                self.sql_operations('insert', self.db_connect, 'userinfo', **auth_data)
                
                # Initialize scanner components
                self.tel_id = auth_data['tel_id']
                self.bot_token = auth_data['tel_token']
                self.setup_market_streams()
                
                self.auth_window.destroy()
                self.master.title(f"Market Scanner - {auth_data['name']}")
                self.start_scanning()
                
            except ccxt.AuthenticationError:
                self.show_error("Authentication Error", "Invalid API credentials. Please check your API Key and Secret.")
            except Exception as e:
                self.show_error("Connection Error", f"Unable to connect to exchange: {str(e)}")
        else:
            self.show_error("Validation Error", "All fields are required")

    def load_saved_login(self):
        saved_data = self.sql_operations('fetch', self.db_connect, 'userinfo')
        if saved_data:
            last_login = saved_data[-1]
            for key, var in self.auth_vars.items():
                var.set(last_login.get(key, ''))

    def change(self, market_type):
        """Handle market type changes"""
        if market_type == 'USDT':
            return [m for m in self.tickers() if m.endswith('USDT')]
        elif market_type == 'BTC':
            return [m for m in self.tickers() if m.endswith('BTC')]
        return []

    def cleanup(self):
        """Enhanced cleanup with proper checks"""
        # Stop scanning
        self.scanning = False
        
        # Clear active streams
        if hasattr(self, 'active_streams'):
            self.active_streams.clear()
            
        # Close exchange connection
        if hasattr(self, 'binance') and self.binance:
            try:
                self.binance.close()
                self.logger.info("Exchange connection closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing exchange connection: {e}")
                
        # Save any pending data
        if hasattr(self, 'save_favorites'):
            self.save_favorites()
            
        # Close database connections
        if hasattr(self, 'db_connect'):
            self.sql_operations('close', self.db_connect, None)

        
    def create_main_layout(self):
        self.create_navigation_bar()
        self.create_left_sidebar()
        self.create_center_panel()
        self.create_right_panel()
        self.create_bottom_panel()
        
    def create_navigation_bar(self):
        nav_frame = ttk.Frame(self.master, style='Dashboard.TFrame')
        nav_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        
        # Exchange selector
        exchanges = ['Binance', 'CoinEx', 'Okex', 'BingX']
        ttk.OptionMenu(nav_frame, self.exchange_var, 'Binance', *exchanges).pack(side='left', padx=5)
        
        # Timeframe selector
        timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        ttk.OptionMenu(nav_frame, self.timeframe_var, '15m', *timeframes).pack(side='left', padx=5)
        
        # Scanner controls
        ttk.Button(nav_frame, text='Start', command=self.start_scanning).pack(side='left', padx=5)
        ttk.Button(nav_frame, text='Stop', command=self.stop_scanning).pack(side='left', padx=5)
        
        self.status_label = ttk.Label(nav_frame, text="Scanner Status: Stopped", style='Dashboard.TLabel')
        self.status_label.pack(side='right', padx=5)
        
    def create_left_sidebar(self):
        sidebar = ttk.Frame(self.master, style='Dashboard.TFrame')
        sidebar.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        
        # Market search
        ttk.Label(sidebar, text="Market Search", style='Dashboard.TLabel').pack()
        self.market_search = ttk.Entry(sidebar)
        self.market_search.pack(fill='x', pady=2)
        self.market_search.bind('<KeyRelease>', self.filter_markets)
        
        # Market list
        self.market_listbox = tk.Listbox(sidebar, bg='#2C2C2C', fg='white', height=20)
        self.market_listbox.pack(fill='both', expand=True)
        self.market_listbox.bind('<<ListboxSelect>>', self.on_market_select)
        
        # Favorites section
        ttk.Label(sidebar, text="Favorites", style='Dashboard.TLabel').pack(pady=(10,2))
        self.favorites_listbox = tk.Listbox(sidebar, bg='#2C2C2C', fg='white', height=10)
        self.favorites_listbox.pack(fill='x')
        
        # Favorites controls
        controls = ttk.Frame(sidebar, style='Dashboard.TFrame')
        controls.pack(fill='x')
        ttk.Button(controls, text="Add", command=self.add_to_favorites).pack(side='left', padx=2)
        ttk.Button(controls, text="Remove", command=self.remove_from_favorites).pack(side='right', padx=2)
        
    def create_center_panel(self):
        center = ttk.Frame(self.master, style='Dashboard.TFrame')
        center.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        
        # Price chart
        fig = mpf.figure(style='charles')
        self.chart_canvas = FigureCanvasTkAgg(fig, master=center)
        self.chart_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Technical indicators
        indicators = ttk.Frame(center, style='Dashboard.TFrame')
        indicators.pack(fill='x')
        self.rsi_label = ttk.Label(indicators, text="RSI: --", style='Dashboard.TLabel')
        self.rsi_label.pack(side='left', padx=5)
        self.macd_label = ttk.Label(indicators, text="MACD: --", style='Dashboard.TLabel')
        self.macd_label.pack(side='left', padx=5)
        
    def create_right_panel(self):
        right = ttk.Frame(self.master, style='Dashboard.TFrame')
        right.grid(row=1, column=2, sticky='nsew', padx=5, pady=5)
        
        # Signals panel
        signals = ttk.LabelFrame(right, text="Active Signals")
        signals.pack(fill='x', padx=5, pady=5)
        
        columns = ('Time', 'Market', 'Type', 'Price', 'Action')
        self.signals_tree = ttk.Treeview(signals, columns=columns, show='headings', height=10)
        for col in columns:
            self.signals_tree.heading(col, text=col)
        self.signals_tree.pack(fill='x')
        
        # Alerts panel
        alerts = ttk.LabelFrame(right, text="Alerts")
        alerts.pack(fill='x', padx=5, pady=5)
        self.alerts_text = tk.Text(alerts, height=10, bg='#2C2C2C', fg='white')
        self.alerts_text.pack(fill='x')
        
    def create_bottom_panel(self):
        bottom = ttk.Frame(self.master, style='Dashboard.TFrame')
        bottom.grid(row=2, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        
        # Performance metrics
        self.win_rate_label = ttk.Label(bottom, text="Win Rate: --", style='Dashboard.TLabel')
        self.win_rate_label.pack(side='left', padx=10)
        
        self.profit_factor_label = ttk.Label(bottom, text="Profit Factor: --", style='Dashboard.TLabel')
        self.profit_factor_label.pack(side='left', padx=10)
        
        # Export buttons
        ttk.Button(bottom, text="Export Data", command=self.export_to_excel).pack(side='right', padx=5)
        ttk.Button(bottom, text="Export Signals", command=self.export_signals).pack(side='right', padx=5)


    def filter_markets(self, event=None):
        search_term = self.market_search.get().upper()
        self.market_listbox.delete(0, tk.END)
        markets = self.tickers()
        filtered_markets = [market for market in markets if search_term in market]
        for market in filtered_markets:
            self.market_listbox.insert(tk.END, market)

    def on_market_select(self, event=None):
        selection = self.market_listbox.curselection()
        if selection:
            market = self.market_listbox.get(selection[0])
            self.current_market = market
            self.update_market_displays()

    def update_market_displays(self):
        self.update_price_chart()
        self.update_indicators()
        self.update_signals_display()

    def update_price_chart(self):
        if self.current_market:
            df = self.fetch_market_data(self.current_market, self.timeframe_var.get())
            if df is not None:
                fig = mpf.figure(style='charles')
                ax = fig.add_subplot(1,1,1)
                mpf.plot(df, type='candle', style='charles', ax=ax)
                self.chart_canvas.figure = fig
                self.chart_canvas.draw()

    def update_indicators(self):
        if self.current_market:
            df = self.fetch_market_data(self.current_market, self.timeframe_var.get())
            if df is not None:
                rsi = talib.RSI(df['close']).iloc[-1]
                macd, signal, _ = talib.MACD(df['close'])
                self.rsi_label.config(text=f"RSI: {rsi:.2f}")
                self.macd_label.config(text=f"MACD: {macd.iloc[-1]:.2f}")

    def update_signals_display(self):
        for item in self.signals_tree.get_children():
            self.signals_tree.delete(item)
        signals = self.sql_operations('fetch', self.db_signals, 'Signals')
        for signal in signals[-10:]:
            self.signals_tree.insert('', 'end', values=(
                signal['timestamp'],
                signal['market'],
                signal['signal_type'],
                signal['price'],
                'Monitor'
            ))

    def add_to_favorites(self):
        selection = self.market_listbox.curselection()
        if selection:
            market = self.market_listbox.get(selection[0])
            if market not in self.get_favorites():
                self.favorites_listbox.insert(tk.END, market)
                self.save_favorites()

    def remove_from_favorites(self):
        selection = self.favorites_listbox.curselection()
        if selection:
            self.favorites_listbox.delete(selection[0])
            self.save_favorites()

    def get_favorites(self):
        return [self.favorites_listbox.get(i) for i in range(self.favorites_listbox.size())]

    def save_favorites(self):
        favorites = self.get_favorites()
        with open('config/favorites.json', 'w') as f:
            json.dump(favorites, f)

    def load_favorites(self):
        try:
            with open('config/favorites.json', 'r') as f:
                favorites = json.load(f)
                for market in favorites:
                    self.favorites_listbox.insert(tk.END, market)
        except FileNotFoundError:
            pass

    def add_alert(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.alerts_text.insert('1.0', f"[{timestamp}] {message}\n")
        self.alerts_text.see('1.0')

    def start_scanning(self):
        if not self.scanning:
            self.scanning = True
            self.status_label.config(text="Scanner Status: Running", foreground='green')
            threading.Thread(target=self.scan_for_signals, daemon=True).start()

    def stop_scanning(self):
        self.scanning = False
        self.status_label.config(text="Scanner Status: Stopped", foreground='red')

    def export_to_excel(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"market_data_{timestamp}.xlsx"
        
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        
        signals_df = pd.DataFrame(self.sql_operations('fetch', self.db_signals, 'Signals'))
        signals_df.to_excel(writer, sheet_name='Signals', index=False)
        
        if hasattr(self, 'current_market'):
            market_data = self.fetch_market_data(self.current_market, self.timeframe_var.get())
            if market_data is not None:
                market_data.to_excel(writer, sheet_name='Market Data', index=True)
        
        writer.close()
        self.logger.info(f"Data exported to {filename}")

    def export_signals(self):
        signals = self.sql_operations('fetch', self.db_signals, 'Signals')
        if signals:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"signals_{timestamp}.csv"
            
            df = pd.DataFrame(signals)
            df.to_csv(filename, index=False)
            self.logger.info(f"Signals exported to {filename}")

    def update_performance_metrics(self):
        signals = self.sql_operations('fetch', self.db_signals, 'Signals')
        if signals:
            win_rate = self.calculate_win_rate()
            profit_factor = self.calculate_profit_factor()
            
            self.win_rate_label.config(text=f"Win Rate: {win_rate:.1%}")
            self.profit_factor_label.config(text=f"Profit Factor: {profit_factor:.2f}")

    def calculate_win_rate(self):
        signals = self.sql_operations('fetch', self.db_signals, 'Signals')
        if not signals:
            return 0
        successful = sum(1 for s in signals if s.get('result') == 'success')
        return successful / len(signals)

    def calculate_profit_factor(self):
        signals = self.sql_operations('fetch', self.db_signals, 'Signals')
        if not signals:
            return 0
        wins = sum(s['profit'] for s in signals if s.get('profit', 0) > 0)
        losses = abs(sum(s['profit'] for s in signals if s.get('profit', 0) < 0))
        return wins / losses if losses > 0 else 0

    def sql_operations(self, operation, db, table, **kwargs):
        conn = sqlite3.connect(db)
        c = conn.cursor()
        
        try:
            if operation == 'create':
                if table == 'userinfo':
                    c.execute('''CREATE TABLE IF NOT EXISTS userinfo
                                (name TEXT, key TEXT, secret TEXT, 
                                 phrase TEXT, tel_id TEXT, tel_token TEXT)''')
                elif table == 'Signals':
                    c.execute('''CREATE TABLE IF NOT EXISTS Signals
                                (market TEXT, timeframe TEXT, signal_type TEXT,
                                 price REAL, volume_trend TEXT, vwap REAL,
                                 rsi REAL, timestamp TEXT)''')
                                 
            elif operation == 'insert':
                placeholders = ', '.join(['?' for _ in kwargs])
                columns = ', '.join(kwargs.keys())
                values = tuple(kwargs.values())
                c.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", values)
                
            elif operation == 'fetch':
                c.execute(f"SELECT * FROM {table}")
                columns = [description[0] for description in c.description]
                return [dict(zip(columns, row)) for row in c.fetchall()]
                
            elif operation == 'delete':
                c.execute(f"DELETE FROM {table}")
                
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Database error: {e}")
            
        finally:
            conn.close()

    def setup_market_streams(self):
        """Initialize market data streams"""
        self.market_data = {}
        self.active_streams = set()
        
        if hasattr(self, 'binance'):
            markets = self.tickers()
            for market in markets[:10]:  # Start with top 10 markets
                self.start_market_stream(market)

    def start_market_stream(self, market):
        """Start streaming data for a specific market"""
        if market not in self.active_streams:
            self.active_streams.add(market)
            threading.Thread(target=self.stream_market_data, 
                          args=(market,), daemon=True).start()

    def stream_market_data(self, market):
        """Stream market data and update internal storage"""
        while self.scanning and market in self.active_streams:
            try:
                ticker = self.binance.fetch_ticker(market)
                self.market_data[market] = {
                    'price': float(ticker['last']),
                    'volume': float(ticker['quoteVolume']),
                    'change': float(ticker['percentage']),
                    'timestamp': ticker['timestamp']
                }
                time.sleep(1)  # Rate limiting
            except Exception as e:
                self.logger.error(f"Stream error for {market}: {e}")
                time.sleep(5)  # Error cooldown

    def cleanup(self):
        """Cleanup resources before exit"""
        self.scanning = False
        self.active_streams.clear()
        if hasattr(self, 'binance'):
            self.binance.close()

    def run(self):
        """Main application loop"""
        try:
            self.master.mainloop()
        except Exception as e:
            self.logger.error(f"Application error: {e}")
        finally:
            self.cleanup()

    def on_closing(self):
        """Handle window closing event"""
        self.cleanup()
        self.master.destroy()

    def initialize_exchange(self):
        """Initialize exchange connection with proper configuration"""
        try:
            self.binance = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'recvWindow': 5000
                },
                'timeout': 30000
            })
            
            # Load markets
            self.binance.load_markets()
            self.logger.info("Exchange connection initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Exchange initialization error: {e}")
            return False


    def cleanup(self):
        """Enhanced cleanup with proper checks"""
        self.scanning = False
        
        if hasattr(self, 'active_streams'):
            self.active_streams.clear()
            
        if hasattr(self, 'binance'):
            try:
                # Proper way to close ccxt exchange
                if isinstance(self.binance, ccxt.binance):
                    self.binance = None
                self.logger.info("Exchange connection closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing exchange connection: {e}")
                
        if hasattr(self, 'save_favorites'):
            self.save_favorites()
            
        if hasattr(self, 'db_connect'):
            self.sql_operations('close', self.db_connect, None)

    def tickers(self):
        """Fetch available market tickers"""
        try:
            markets = self.binance.fetch_tickers()
            return [m for m in markets.keys() if m.endswith('USDT')]
        except Exception as e:
            self.logger.error(f"Error fetching tickers: {e}")
            return []

    def update_market_overview(self):
        if hasattr(self, 'current_market'):
            try:
                ticker = self.binance.fetch_ticker(self.current_market)
                
                # Update price label
                price = float(ticker['last'])
                self.price_label.configure(text=f"Price: {price:.8f}")
                
                # Update 24h change
                change = float(ticker['percentage'])
                color = 'green' if change > 0 else 'red'
                self.change_label.configure(text=f"24h Change: {change:.2f}%", foreground=color)
                
                # Update volume
                volume = float(ticker['quoteVolume'])
                self.volume_label.configure(text=f"Volume: {volume:,.0f} USDT")
                
            except Exception as e:
                self.logger.error(f"Market overview update error: {e}")

    def update_performance_dashboard(self):
        signals = self.sql_operations('fetch', self.db_signals, 'Signals')
        if signals:
            # Calculate metrics
            total_signals = len(signals)
            win_rate = self.calculate_win_rate()
            profit_factor = self.calculate_profit_factor()
            
            # Update labels
            self.signals_count_label.configure(text=f"Total Signals: {total_signals}")
            self.win_rate_label.configure(text=f"Win Rate: {win_rate:.1%}")
            self.profit_factor_label.configure(text=f"Profit Factor: {profit_factor:.2f}")

    def check_rate_limit(self):
        """Check and handle API rate limits"""
        current_time = time.time()
        if current_time - self.rate_limits['last_reset'] >= self.rate_limits['cooldown_period']:
            self.rate_limits['api_calls'] = 0
            self.rate_limits['last_reset'] = current_time
        
        if self.rate_limits['api_calls'] >= self.rate_limits['max_calls_per_minute']:
            sleep_time = self.rate_limits['cooldown_period'] - (current_time - self.rate_limits['last_reset'])
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.rate_limits['api_calls'] = 0
            self.rate_limits['last_reset'] = time.time()
        
        self.rate_limits['api_calls'] += 1

    def update_gui(self):
        """Update all GUI components"""
        if self.scanning:
            self.update_market_overview()
            self.update_signals_display()
            self.update_performance_dashboard()
            self.master.after(1000, self.update_gui)

    def validate_signal(self, signal):
        """Validate signal against multiple criteria"""
        if not signal:
            return False
            
        # Volume validation
        volume_valid = self.validate_volume(signal['market'])
        
        # Price action validation
        price_valid = self.validate_price_action(signal['market'])
        
        # Technical validation
        tech_valid = self.validate_technical_indicators(signal['market'])
        
        return all([volume_valid, price_valid, tech_valid])

    def validate_volume(self, market):
        """Check if volume meets minimum requirements"""
        try:
            ticker = self.binance.fetch_ticker(market)
            volume = float(ticker['quoteVolume'])
            return volume >= 1000000  # Minimum 1M USDT volume
        except Exception as e:
            self.logger.error(f"Volume validation error: {e}")
            return False

    def validate_price_action(self, market):
        """Validate price action patterns"""
        df = self.fetch_market_data(market, '1h')
        if df is not None:
            # Check for strong candlestick patterns
            last_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            # Bullish engulfing
            if (last_candle['close'] > last_candle['open'] and
                prev_candle['close'] < prev_candle['open'] and
                last_candle['close'] > prev_candle['open'] and
                last_candle['open'] < prev_candle['close']):
                return True
                
            # Other patterns can be added here
            
        return False

    def validate_technical_indicators(self, market):
        """Validate technical indicators"""
        df = self.fetch_market_data(market, '1h')
        if df is not None:
            # Calculate indicators
            rsi = talib.RSI(df['close']).iloc[-1]
            macd, signal, _ = talib.MACD(df['close'])
            
            # Define conditions
            rsi_oversold = rsi < 30
            macd_bullish = macd.iloc[-1] > signal.iloc[-1]
            
            return rsi_oversold or macd_bullish
            
        return False

def main():
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", lambda: ScannerGUI(root).on_closing())
    app = ScannerGUI(root)
    app.run()

if __name__ == "__main__":
    main()

