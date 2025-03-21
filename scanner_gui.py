import tkinter as tk
from tkinter import ttk
from base import BaseScanner
import ccxt
import sqlite3
from datetime import datetime
import time
import csv
from market_app import MarketApp
import talib
import pandas as pd
from tkinter import messagebox 
from market_scanner import MarketScanner
import threading
import logging

class ScannerGUI(MarketScanner):
    def __init__(self, master=None):
        super().__init__(master)
        
        # Database initialization
        self.db_connect = "connection.db"
        self.db_signals = "Signals.db"
        
        # Create both database tables
        self.sql_operations('create', self.db_connect, 'userinfo')
        self.sql_operations('create', self.db_signals, 'Signals')
        
        # Core components initialization
        self.tel_id = None
        self.bot_tocken = None
        self.scanning = False
        self.filtered_list = []
        self.selected_markets = []
        self.timeframesdic = {
            '1s': 0, '1m': 1, '5m': 2, '15m': 3,
            '30m': 4, '1h': 5, '4h': 6, '1d': 7
        }
        
        # Setup GUI and authentication
        self.setup_gui()
        self.auth = self.user_auth(self.db_connect, "userinfo")
    def user_auth(self, file_name, table_name):
        self.auth = tk.Toplevel()
        self.auth.title('User Authentication')
        self.auth.geometry('500x300')
        self.auth.resizable(True, True)
        self.auth.columnconfigure(0, weight=1)

        username_label = tk.Label(self.auth, text='Username:')
        username_label.grid(row=0, column=0, padx=5, pady=5)
        self.username_entry = tk.Entry(self.auth, width=50)
        self.username_entry.grid(row=0, column=1, padx=5, pady=5)

        private_key_label = tk.Label(self.auth, text='Private Key:')
        private_key_label.grid(row=1, column=0, padx=5, pady=5)
        self.private_key_entry = tk.Entry(self.auth, show='*', width=50)
        self.private_key_entry.grid(row=1, column=1, padx=5, pady=5)

        sec_key_label = tk.Label(self.auth, text='Secret Key:')
        sec_key_label.grid(row=2, column=0, padx=5, pady=5)
        self.sec_key_entry = tk.Entry(self.auth, show='*', width=50)
        self.sec_key_entry.grid(row=2, column=1, padx=5, pady=5)

        phrase_label = tk.Label(self.auth, text='Phrase:')
        phrase_label.grid(row=3, column=0, padx=5, pady=5)
        self.phrase_entry = tk.Entry(self.auth, show='*', width=50)
        self.phrase_entry.grid(row=3, column=1, padx=5, pady=5)

        tel_id_label = tk.Label(self.auth, text='Tel_Id:')
        tel_id_label.grid(row=4, column=0, padx=5, pady=5)
        self.tel_id_entry = tk.Entry(self.auth, show='*', width=50)
        self.tel_id_entry.grid(row=4, column=1, padx=5, pady=5)

        tocken_label = tk.Label(self.auth, text='Tel_Bot_token')
        tocken_label.grid(row=5, column=0, padx=5, pady=5)
        self.tocken_entry = tk.Entry(self.auth, show='*', width=50)
        self.tocken_entry.grid(row=5, column=1, padx=5, pady=5)

        submit_button = tk.Button(self.auth, text='Submit', width=20, command=self.submit)
        submit_button.grid(row=6, column=0, padx=5, pady=5)

        ok_button = tk.Button(self.auth, text='Cancel', width=20, command=self.quittheapp)
        ok_button.grid(row=6, column=1, padx=5, pady=5)

        return self.auth

    def submit(self):
        try:
            db1 = sqlite3.connect('connection.db')
            c1 = db1.cursor()
            c1.execute('SELECT * FROM userinfo')
            data = c1.fetchone()

            if data:
                private_key = data[1]
                secret_key = data[2]
                phraseword = data[3]
                tel_id = data[4]
                bot_token = data[5]

                if self.con(file_name='connection.db'):
                    self.api_key = str(private_key)
                    self.secret_key = str(secret_key)
                    self.phrase = phraseword
                    self.valid = True
                    self.tel_id = tel_id
                    self.bot_token = bot_token
                else:
                    self.master.destroy()

                if self.auth:
                    try:
                        self.auth.destroy()
                    except Exception as e:
                        print(f"Error destroying auth window: {e}")

            elif data is None and len(self.username_entry.get()) > 0 and len(self.private_key_entry.get()) > 0 and len(self.phrase_entry.get()) > 0:
                username = self.username_entry.get()
                private_key = self.private_key_entry.get()
                secret_key = self.sec_key_entry.get()
                phrase_key = self.phrase_entry.get()
                tel_id = self.tel_id_entry.get()
                bot_token = self.tocken_entry.get()

                c1.execute('INSERT INTO userinfo (name, key , secret ,phrase , tel_id , tel_token ) VALUES ( ?, ?, ?, ?, ?, ?)',
                        (username, private_key, secret_key, phrase_key, tel_id, bot_token))
                db1.commit()

                if self.auth:
                    try:
                        self.auth.destroy()
                    except Exception as e:
                        print(f"Error destroying auth window: {e}")

                if self.con(file_name='connection.db'):
                    self.api_key = str(private_key)
                    self.secret_key = str(secret_key)
                    self.phrase = phrase_key
                    self.valid = True
                    self.tel_id = tel_id
                    self.bot_token = bot_token
                else:
                    self.master.destroy()

            else:
                messagebox.showerror('Invalid User Data', 'Please check Your Data and try again ')
                self.master.destroy()

        except sqlite3.Error as e:
            print(f"Database error: {e}")
            messagebox.showerror('Database Error', 'An error occurred while accessing the database.')

        except Exception as e:
            print(f"Error in submit: {e}")
            messagebox.showerror('Error', 'An unexpected error occurred.')

        finally:
            if db1:
                db1.close()

    def quittheapp(self):
        if self.auth:
            self.auth.destroy()
        self.master.destroy()
    def setup_gui(self):
        # Create main frames for better organization
        self.control_frame = ttk.Frame(self.master)
        self.control_frame.grid(row=0, column=0, columnspan=12, sticky='nsew', padx=10, pady=5)
        
        # Strategy selection in control frame
        str_manu = ['ALL']
        self.choose_str = tk.StringVar(self.master)
        self.choose_str.set('ALL')
        self.option_str = ttk.OptionMenu(self.control_frame, self.choose_str, *str_manu)
        self.option_str.grid(row=0, column=0, padx=5)

        # Timeframes with proper spacing
        self.timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        self.choose_time = tk.StringVar(self.master)
        self.choose_time.set('15m')
        self.option_time = ttk.OptionMenu(self.control_frame, self.choose_time, *self.timeframes)
        self.option_time.grid(row=0, column=1, padx=5)

        # Market options in organized layout
        list_manu = ['Down', 'Top', 'Best-vol', 'Last-vol']
        self.choose_list = tk.StringVar(self.master)
        self.choose_list.set('Best-vol')
        self.option_list = ttk.OptionMenu(self.control_frame, self.choose_list, *list_manu)
        self.option_list.grid(row=0, column=2, padx=5)

        # Exchange selection with consistent spacing
        list_manu_ex = ['Binance', 'CoinEx', 'Okex', 'BingX']
        self.choose_listex = tk.StringVar(self.master)
        self.choose_listex.set('Binance')
        self.option_listex = ttk.OptionMenu(self.control_frame, self.choose_listex, *list_manu_ex)
        self.option_listex.grid(row=0, column=3, padx=5)

        # Control buttons in separate frame
        button_frame = ttk.Frame(self.control_frame)
        button_frame.grid(row=0, column=4, columnspan=2, padx=10)
        
        self.start_button = ttk.Button(button_frame, text='Start Scanning',
                                    command=self.start_scanning)
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(button_frame, text='Stop Scanning',
                                   command=self.stop_scanning)
        self.stop_button.pack(side='left', padx=5)

        # Market selection controls
        market_frame = ttk.Frame(self.master)
        market_frame.grid(row=1, column=0, columnspan=12, sticky='nsew', padx=10, pady=5)
        
        self.user_choice = tk.IntVar(value=0)
        ttk.Checkbutton(market_frame, text='Custom Markets',
                      variable=self.user_choice).pack(side='left', padx=5)
        
        ttk.Button(market_frame, text='Select Markets',
                  command=self.user_choicelist).pack(side='left', padx=5)

        # Status display
        self.status_label = tk.Label(self.master, text="Scanner Status: Stopped",
                                    font=('Arial', 10), fg='red')
        self.status_label.grid(row=3, column=0, columnspan=4)

        # Initialize dashboard
        self.create_enhanced_dashboard()

    def create_enhanced_dashboard(self):
        self.logger.info("Initializing enhanced dashboard...")
        try:
            self.dashboard = ttk.Frame(self.master)
            self.dashboard.grid(row=4, column=0, columnspan=12, sticky='nsew', padx=10, pady=10)
            
            self.create_market_overview_panel()
            self.create_signals_panel()
            self.create_performance_panel()
            self.create_quick_actions()
            
            self.logger.info("All dashboard panels initialized")
        except Exception as e:
            self.logger.error(f"Dashboard creation error: {e}")
    def create_market_overview_panel(self):
        overview = ttk.LabelFrame(self.dashboard, text="Market Overview")
        overview.grid(row=0, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        
        # Market price ticker
        self.price_label = ttk.Label(overview, text="Current Price:", font=('Arial', 12, 'bold'))
        self.price_label.grid(row=0, column=0, padx=5, pady=5)
        
        # 24h Change indicator with color coding
        self.change_label = ttk.Label(overview, text="24h Change:", font=('Arial', 12))
        self.change_label.grid(row=1, column=0, padx=5, pady=5)
        
        # Volume indicator with progress bar
        self.volume_progress = ttk.Progressbar(overview, length=200, mode='determinate')
        self.volume_progress.grid(row=2, column=0, padx=5, pady=5)

    def create_signals_panel(self):
        self.signals_state = {
            'last_update': None,
            'signal_count': 0,
            'active_signals': []
        }
        
        signals = ttk.LabelFrame(self.dashboard, text="Active Signals")
        signals.grid(row=0, column=3, columnspan=3, sticky='nsew', padx=5, pady=5)
        
        # Signal treeview
        columns = ('Time', 'Market', 'Type', 'Strength', 'Action', 'Status')
        self.signals_tree = ttk.Treeview(signals, columns=columns, show='headings', height=8)
        for col in columns:
            self.signals_tree.heading(col, text=col)
            self.signals_tree.column(col, width=100)

    def create_performance_panel(self):
        performance = ttk.LabelFrame(self.dashboard, text="Performance Metrics")
        performance.grid(row=1, column=0, columnspan=6, sticky='nsew', padx=5, pady=5)
        
        # Win rate gauge
        self.win_rate_canvas = tk.Canvas(performance, width=150, height=150, bg='black')
        self.win_rate_canvas.grid(row=0, column=0, padx=5, pady=5)
        
        # Profit chart
        self.profit_canvas = tk.Canvas(performance, width=300, height=150, bg='black')
        self.profit_canvas.grid(row=0, column=1, padx=5, pady=5)

    def create_quick_actions(self):
        actions = ttk.LabelFrame(self.dashboard, text="Quick Actions")
        actions.grid(row=2, column=0, columnspan=6, sticky='nsew', padx=5, pady=5)
        
        # Quick filter buttons
        ttk.Button(actions, text="Top Gainers", command=lambda: self.quick_filter('gainers')).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(actions, text="High Volume", command=lambda: self.quick_filter('volume')).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(actions, text="Strong Signals", command=lambda: self.quick_filter('signals')).grid(row=0, column=2, padx=5, pady=5)

    def monitor_ui_performance(self):
        self.ui_metrics = {
            'update_times': [],
            'signal_processing_times': [],
            'render_times': []
        }
        
        def measure_update_time(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start
                self.ui_metrics['update_times'].append(duration)
                if duration > 0.1:  # Slow update threshold
                    self.logger.warning(f"Slow UI update: {func.__name__} took {duration:.3f}s")
                return result
            return wrapper
        
        # Apply to key methods
        self.update_dashboard = measure_update_time(self.update_dashboard)
    def start_scanning(self):
        if not self.scanning:
            try:
                self.binance = ccxt.binance({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True,
                        'recvWindow': 5000
                    },
                    'timeout': 30000  # Increase timeout to 30 seconds
                })
                
                # Test connection before starting
                self.binance.load_markets()
                
                self.scanning = True
                self.status_label.config(text="Scanner Status: Running", fg='green')
                
                threading.Thread(target=self.scan_for_signals, daemon=True).start()
                self.send_telegram_update("🚀 Scanner Started - Monitoring Markets")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize exchange: {e}")
                self.status_label.config(text="Connection Error", fg='red')


    def stop_scanning(self):
        self.scanning = False
        self.status_label.config(text="Scanner Status: Stopped", fg='red')
        self.send_telegram_update("⏹️ Market Scanner Stopped")

    def update_dashboard(self):
        if self.scanning:
            try:
                self.update_market_overview()
                self.status_label.config(text="Updating market overview...", fg='blue')
                
                self.update_signals_display()
                self.status_label.config(text="Updating signals...", fg='blue')
                
                self.update_performance_metrics()
                self.status_label.config(text="Scanner Status: Running", fg='green')
                
                self.master.after(1000, self.update_dashboard)
            except Exception as e:
                self.logger.error(f"Dashboard update error: {e}")
                self.status_label.config(text="Update error - check logs", fg='red')

    def update_market_overview(self):
        if not hasattr(self, 'price_label'):
            return
            
        try:
            if self.selected_markets:
                market = self.selected_markets[0]
                ticker = self.binance.fetch_ticker(market)
                
                # Use foreground instead of fg for ttk widgets
                self.price_label.configure(text=f"Price: {ticker['last']:.8f}")
                
                change = float(ticker['percentage'])
                color = 'green' if change > 0 else 'red'
                self.change_label.configure(text=f"24h Change: {change:.2f}%", foreground=color)
                
                volume = float(ticker['quoteVolume'])
                max_volume = 100000000  # 100M baseline
                volume_percent = min((volume / max_volume) * 100, 100)
                self.volume_progress['value'] = volume_percent
                
        except Exception as e:
            self.logger.error(f"Error updating market overview: {e}")


    def quick_filter(self, filter_type):
        if filter_type == 'gainers':
            filtered_markets = self.change('Top')
        elif filter_type == 'volume':
            filtered_markets = self.change('Best-vol')
        elif filter_type == 'signals':
            filtered_markets = self.get_strong_signal_markets()
        
        self.selected_markets = filtered_markets
        self.update_selected_markets_display()

    def get_strong_signal_markets(self):
        strong_signals = []
        for market in self.tickers():
            df = self.fetch_market_data(market, self.choose_time.get())
            if df is not None:
                signals = self.generate_signals(df)
                if any(signal.get('strength', '') == 'high' for signal in signals):
                    strong_signals.append(market)
        return strong_signals
    def user_choicelist(self):
        exchange_config = {
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'recvWindow': 5000
            }
        }
        
        self.binance = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        total_list = self.tickers()
        insi = MarketApp(self, total_list)
        insi.run_choiceApp()

    def tickers(self):
        asset = "USDT"
        tickers = self.binance.fetch_tickers()
        filtered_symbols = [symbol for symbol in tickers if symbol.endswith(f"/{asset}")]
        filtered_s = []

        for market in filtered_symbols:
            try:
                usdt_volume = float(tickers[market]['baseVolume']) * float(tickers[market]['last'])
                filtered_s.append((market, usdt_volume))
            except Exception as e:
                continue

        sorted_symbols = sorted(filtered_s, key=lambda symbol: symbol[1], reverse=True)
        sorted_markets = [symbol[0] for symbol in sorted_symbols]
        return sorted_markets

    def change(self, param=None):
        asset = 'USDT'
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                tickers = self.binance.fetch_tickers()
                tic = pd.DataFrame(tickers).transpose()
                tic = tic[tic.index.str.contains(f"{asset}")]
                df = tic.drop(['vwap', 'askVolume', 'previousClose', 'symbol', 'timestamp', 
                            'info', 'quoteVolume', 'datetime', 'bidVolume'], axis=1)
                
                if param == 'Best-vol':
                    df['volch'] = df['baseVolume'] * df['last']
                    df = df[df['volch'] > 10000]
                    df = df.sort_values('volch', ascending=False)
                    return df.head(100).index.tolist()
                    
                elif param == 'Last-vol':
                    df['volch'] = df['baseVolume'] * df['last']
                    df = df.sort_values('volch', ascending=False)
                    return df.head(50).index.tolist()
                    
                elif param == 'Top':
                    df = df.sort_values('percentage', ascending=False)
                    return df.head(100).index.tolist()
                    
                elif param == 'Down':
                    df = df.sort_values('percentage', ascending=True)
                    return df.head(100).index.tolist()

            except Exception as e:
                self.logger.error(f"Error in change method: {e}")
                return []

    def fetch_market_data(self, market, timeframe, limit=100):
        if market.startswith('USDT/'):
            return None
            
        try:
            ohlcv = self.binance.fetch_ohlcv(market, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            self.logger.error(f"Data fetch error for {market}: {e}")
            return None
    def analyze_market(self, market, timeframe):
        try:
            df = self.fetch_market_data(market, timeframe)
            if df is not None and not df.empty:
                # Calculate technical indicators
                df['rsi'] = talib.RSI(df['close'])
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
                df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
                df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
                
                volume_profile = self.analyze_volume_profile(df)
                signals = self.generate_signals(df)
                
                if volume_profile and signals:
                    for signal in signals:
                        signal['volume_context'] = volume_profile['volume_trend']
                        signal['vwap'] = volume_profile['vwap']
                
                if signals:
                    for signal in signals:
                        message = (
                            f"🔔 Signal Alert!\n"
                            f"Market: {market}\n"
                            f"Signal: {signal['type']}\n"
                            f"Price: {signal['price']}\n"
                            f"VWAP: {signal.get('vwap', 'N/A')}\n"
                            f"Volume Trend: {signal.get('volume_context', 'N/A')}\n"
                            f"RSI: {df['rsi'].iloc[-1]:.2f}\n"
                            f"Timeframe: {timeframe}"
                        )
                        self.send_telegram_update(message)
                        
                        self.sql_operations('insert', self.db_signals, 'Signals',
                                        market=market,
                                        timeframe=timeframe,
                                        signal_type=signal['type'],
                                        price=signal['price'],
                                        volume_trend=signal.get('volume_context', ''),
                                        vwap=signal.get('vwap', 0.0),
                                        rsi=df['rsi'].iloc[-1],
                                        timestamp=str(datetime.now()))
                        
        except Exception as e:
            self.logger.error(f"Error analyzing market {market}: {e}")
    def update_signals_display(self):
        try:
            if hasattr(self, 'signals_tree'):
                # Clear existing items
                for item in self.signals_tree.get_children():
                    self.signals_tree.delete(item)
                
                # Fetch recent signals from database
                signals = self.sql_operations('fetch', self.db_signals, 'Signals')
                
                for signal in signals[-10:]:  # Show last 10 signals
                    self.signals_tree.insert('', 'end', values=(
                        signal['timestamp'],
                        signal['market'],
                        signal['signal_type'],
                        signal.get('strength', 'N/A'),
                        'Monitor',
                        'Active'
                    ))
        except Exception as e:
            self.logger.error(f"Error updating signals display: {e}")

    def update_performance_metrics(self):
        try:
            if hasattr(self, 'win_rate_canvas'):
                # Clear canvas
                self.win_rate_canvas.delete("all")
                
                # Calculate win rate from signals
                signals = self.sql_operations('fetch', self.db_signals, 'Signals')
                if signals:
                    win_rate = 0.65  # Example win rate
                    self.draw_gauge(win_rate)
                    
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
            
    def draw_gauge(self, value):
        # Draw win rate gauge visualization
        canvas = self.win_rate_canvas
        width = 150
        height = 150
        
        # Draw arc
        angle = value * 180
        canvas.create_arc(10, 10, width-10, height-10, 
                        start=180, extent=-angle,
                        fill='green' if value > 0.5 else 'red')
                        
        # Draw value text
        canvas.create_text(width/2, height/2,
                        text=f"{value*100:.1f}%",
                        fill='white',
                        font=('Arial', 16, 'bold'))

    def update_selected_markets_display(self):
        try:
            # Clear and update markets listbox
            if hasattr(self, 'selected_markets_listbox'):
                self.selected_markets_listbox.delete(0, tk.END)
                for market in self.selected_markets:
                    self.selected_markets_listbox.insert(tk.END, market)
                    
            # Update market count label if exists
            if hasattr(self, 'market_count_label'):
                count = len(self.selected_markets)
                self.market_count_label.config(text=f"Selected Markets: {count}")
                
        except Exception as e:
            self.logger.error(f"Error updating markets display: {e}")


    def analyze_price_action(self, df):
        if df is None or df.empty:
            return {'trend': 'neutral'}
            
        try:
            df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
            df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            if 'USDT/BRL' in str(df):
                return {
                    'trend': 'neutral',
                    'volatility': 0,
                    'trend_strength': 0,
                    'momentum': 0
                }
                
            current_price = df['close'].iloc[-1]
            trend = 'neutral'
            
            if not df['ema_20'].isna().all() and not df['ema_50'].isna().all():
                ema20 = df['ema_20'].iloc[-1]
                ema50 = df['ema_50'].iloc[-1]
                trend = 'uptrend' if current_price > ema20 > ema50 else 'downtrend' if current_price < ema20 < ema50 else 'neutral'
            
            return {
                'trend': trend,
                'volatility': df['volatility'].iloc[-1] if not df['volatility'].isna().all() else 0,
                'trend_strength': self.detect_trend_strength(df),
                'momentum': df['returns'].iloc[-5:].mean() if not df['returns'].isna().all() else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error in analyze_price_action: {e}")
            return {
                'trend': 'neutral',
                'volatility': 0,
                'trend_strength': 0,
                'momentum': 0
            }

    def monitor_price_action(self):
        while self.scanning:
            for market in self.selected_markets:
                df = self.fetch_market_data(market, self.choose_time.get())
                if self.detect_signal(df):
                    self.process_signal(market, df)
            time.sleep(self.rate_config['window'])

    def filter_markets(self, markets):
        filtered = []
        for market in markets:
            df = self.fetch_market_data(market, self.choose_time.get())
            if df is not None:
                volume = df['volume'].mean()
                volatility = df['close'].pct_change().std()
                
                if volume > self.min_volume and volatility > self.min_volatility:
                    filtered.append(market)
        return filtered

def main():
    root = tk.Tk()
    app = ScannerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
