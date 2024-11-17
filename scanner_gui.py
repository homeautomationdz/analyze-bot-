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
from tkinter import messagebox, ttk
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

                if self.con(file_name='connection.db'):  # Added file_name parameter here
                    self.api_key = str(private_key)
                    self.secret_key = str(secret_key)
                    self.phrase = phrase_key  # Changed phraseword to phrase_key
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
        # Strategy selection
        str_manu = ['ALL']
        self.choose_str = tk.StringVar(self.master)
        self.choose_str.set('ALL')
        self.option_str = tk.OptionMenu(self.master, self.choose_str, *str_manu)
        self.option_str.configure(font="Arial,10", state='normal')
        self.option_str.grid(row=0, column=0)

        # Timeframes
        self.timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        self.choose_time = tk.StringVar(self.master)
        self.choose_time.set('15m')
        self.option_time = tk.OptionMenu(self.master, self.choose_time, *self.timeframes)
        self.option_time.configure(font="Arial,10", state='normal')
        self.option_time.grid(row=1, column=3)

        # Market list options
        list_manu = ['Down', 'Top', 'Best-vol', 'Last-vol']
        self.choose_list = tk.StringVar(self.master)
        self.choose_list.set('Best-vol')
        self.option_list = tk.OptionMenu(self.master, self.choose_list, *list_manu)
        self.option_list.configure(font="Arial,10", state='normal')
        self.option_list.grid(row=2, column=0)

        # Exchange selection
        list_manu_ex = ['Binance', 'CoinEx', 'Okex', 'BingX']
        self.choose_listex = tk.StringVar(self.master)
        self.choose_listex.set('Binance')
        self.option_listex = tk.OptionMenu(self.master, self.choose_listex, *list_manu_ex)
        self.option_listex.configure(font="Arial,10", state='normal')
        self.option_listex.grid(row=2, column=2)

        # Control buttons
        self.start_button = tk.Button(self.master, text='Start Scanning',
                                    background='green', font=('Arial', 10),
                                    command=self.start_scanning)
        self.start_button.grid(row=0, column=7)
        
        self.stop_button = tk.Button(self.master, text='Stop Scanning',
                                   background='red', font=('Arial', 10),
                                   command=self.stop_scanning)
        self.stop_button.grid(row=0, column=8)

        # User choice checkbox
        self.user_choice = tk.IntVar(value=0)
        user_choice = tk.Checkbutton(self.master, text='Custom Markets',
                                   font=('Arial', 10), fg='red',
                                   variable=self.user_choice)
        user_choice.grid(row=2, column=10)

        self.button_choice = tk.Button(self.master, text='Select Markets',
                                     background='gray', font=('Arial', 10),
                                     command=self.user_choicelist)
        self.button_choice.grid(row=2, column=9)

        # Status label
        self.status_label = tk.Label(self.master, text="Scanner Status: Stopped",
                                   font=('Arial', 10), fg='red')
        self.status_label.grid(row=3, column=0, columnspan=4)

    def create_advanced_visualization_panel(self):
        viz_frame = ttk.LabelFrame(self.dashboard_frame, text="Advanced Visualization")
        viz_frame.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky='nsew')
        
        # Market Depth Visualization
        self.create_depth_chart(viz_frame)
        
        # Volume Profile Analysis
        self.create_volume_profile_chart(viz_frame)
        
        # Order Flow Heatmap
        self.create_order_flow_heatmap(viz_frame)
        
        return viz_frame

    def create_depth_chart(self, parent):
        depth_frame = ttk.Frame(parent)
        depth_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        
        self.depth_canvas = tk.Canvas(depth_frame, width=400, height=300, bg='black')
        self.depth_canvas.grid(row=0, column=0, sticky='nsew')
        
        # Add controls
        ttk.Button(depth_frame, text="Update Depth", 
                command=self.update_depth_chart).grid(row=1, column=0)

    def create_volume_profile_chart(self, parent):
        profile_frame = ttk.Frame(parent)
        profile_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        
        self.profile_canvas = tk.Canvas(profile_frame, width=400, height=300, bg='black')
        self.profile_canvas.grid(row=0, column=0, sticky='nsew')

    def create_order_flow_heatmap(self, parent):
        heatmap_frame = ttk.Frame(parent)
        heatmap_frame.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')
        
        self.heatmap_canvas = tk.Canvas(heatmap_frame, width=400, height=300, bg='black')
        self.heatmap_canvas.grid(row=0, column=0, sticky='nsew')
        self.create_advanced_visualization_panel()


    def create_advanced_market_analysis_panel(self):
        analysis_frame = ttk.LabelFrame(self.dashboard_frame, text="Advanced Market Analysis")
        analysis_frame.grid(row=6, column=0, columnspan=3, padx=5, pady=5, sticky='nsew')
        
        # Order Flow Display
        self.create_order_flow_display(analysis_frame)
        
        # Institutional Activity Display
        self.create_institutional_display(analysis_frame)
        
        # Volume Profile Display
        self.create_volume_profile_display(analysis_frame)
        
        return analysis_frame
    def create_order_flow_display(self, parent):
        order_flow_frame = ttk.LabelFrame(parent, text="Order Flow Analysis")
        order_flow_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        
        # Create order flow visualization
        self.order_flow_canvas = tk.Canvas(order_flow_frame, width=300, height=200)
        self.order_flow_canvas.grid(row=0, column=0, sticky='nsew')
        
        # Add controls
        ttk.Button(order_flow_frame, text="Update", 
                command=self.update_order_flow).grid(row=1, column=0)
    def update_market_analysis(self):
        if self.scanning:
            market = self.get_selected_market()
            timeframe = self.choose_time.get()
            
            # Update order flow analysis
            order_flow = self.analyze_enhanced_order_flow(market)
            self.update_order_flow_display(order_flow)
            
            # Update institutional activity
            inst_activity = self.detect_institutional_activity(market)
            self.update_institutional_display(inst_activity)
            
            # Update volume profile
            volume_profile = self.enhanced_volume_profile(market, timeframe)
            self.update_volume_profile_display(volume_profile)
            
            # Generate alerts for significant events
            self.check_and_alert_significant_events(order_flow, inst_activity, volume_profile)
            
            # Schedule next update
            self.master.after(5000, self.update_market_analysis)
    def check_and_alert_significant_events(self, order_flow, inst_activity, volume_profile):
        alerts = []
        
        # Check for significant order flow events
        if order_flow['liquidity_analysis']['liquidity_score'] > 0.8:
            alerts.append("🔵 High Liquidity Event Detected")
        
        # Check for institutional activity
        if inst_activity['large_orders']:
            alerts.append("🟣 Large Institutional Orders Detected")
        
        # Check for volume profile events
        if volume_profile['trading_opportunities']:
            alerts.append("🟢 Volume Profile Trading Opportunity")
        
        # Send alerts
        for alert in alerts:
            self.send_telegram_update(f"{alert}\nMarket: {self.get_selected_market()}")

    def draw_order_flow_bars(self, liquidity_data):
        width = self.order_flow_canvas.winfo_width()
        height = self.order_flow_canvas.winfo_height()
        
        for cluster in liquidity_data['bid_clusters']:
            y = self.price_to_y(cluster['price'], height)
            w = self.volume_to_width(cluster['volume'], width)
            self.order_flow_canvas.create_rectangle(
                0, y, w, y+2, 
                fill='green', outline='')
        
        for cluster in liquidity_data['ask_clusters']:
            y = self.price_to_y(cluster['price'], height)
            w = self.volume_to_width(cluster['volume'], width)
            self.order_flow_canvas.create_rectangle(
                width-w, y, width, y+2, 
                fill='red', outline='')

    def price_to_y(self, price, height):
        price_range = self.get_price_range()
        if not price_range['max'] - price_range['min']:
            return 0
        return height * (1 - (price - price_range['min']) / (price_range['max'] - price_range['min']))

    def volume_to_width(self, volume, max_width):
        max_volume = self.get_max_volume()
        if not max_volume:
            return 0
        return (volume / max_volume) * max_width * 0.45

    def get_price_range(self):
        market = self.get_selected_market()
        timeframe = self.choose_time.get()
        df = self.fetch_market_data(market, timeframe)
        
        return {
            'max': df['high'].max(),
            'min': df['low'].min()
        }

    def get_max_volume(self):
        market = self.get_selected_market()
        timeframe = self.choose_time.get()
        df = self.fetch_market_data(market, timeframe)
        return df['volume'].max()

    def setup_advanced_filters(self):
        self.filter_config = {
            'volume_threshold': 100000,
            'min_volatility': 0.02,
            'min_pattern_quality': 0.7,
            'correlation_threshold': 0.7
        }
    def setup_performance_dashboard(self):
        self.dashboard_metrics = {
            'trade_performance': {
                'win_rate': self.calculate_win_rate(),
                'profit_factor': self.calculate_profit_factor(),
                'sharpe_ratio': self.calculate_sharpe_ratio()
            },
            'market_analysis': {
                'regime': self.detect_market_regime(),
                'volatility': self.calculate_volatility(),
                'correlation': self.analyze_correlations()
            },
            'risk_metrics': {
                'max_drawdown': self.calculate_drawdown(),
                'value_at_risk': self.calculate_var(),
                'position_exposure': self.calculate_exposure()
            }
        }

    def integrate_gui_components(self):
        # Create main dashboard frame
        self.dashboard_frame = ttk.Frame(self.master)
        self.dashboard_frame.grid(row=4, column=0, columnspan=12, sticky='nsew')
        
        # Add performance metrics display
        self.metrics_display = self.create_metrics_display()
        
        # Add market regime indicator
        self.regime_indicator = self.create_regime_indicator()
        
        # Add trade journal display
        self.journal_display = self.create_journal_display()
        
        # Add risk metrics panel
        self.risk_panel = self.create_risk_panel()
        
        # Update all displays
        self.schedule_updates()
    def create_metrics_display(self):
        metrics_frame = ttk.LabelFrame(self.dashboard_frame, text="Performance Metrics")
        metrics_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        
        # Win Rate
        self.win_rate_label = ttk.Label(metrics_frame, text="Win Rate: ")
        self.win_rate_label.grid(row=0, column=0, padx=5, pady=2)
        self.win_rate_value = ttk.Label(metrics_frame, text="0%")
        self.win_rate_value.grid(row=0, column=1, padx=5, pady=2)
        
        # Profit Factor
        self.profit_factor_label = ttk.Label(metrics_frame, text="Profit Factor: ")
        self.profit_factor_label.grid(row=1, column=0, padx=5, pady=2)
        self.profit_factor_value = ttk.Label(metrics_frame, text="0")
        self.profit_factor_value.grid(row=1, column=1, padx=5, pady=2)
        
        return metrics_frame

    def create_regime_indicator(self):
        regime_frame = ttk.LabelFrame(self.dashboard_frame, text="Market Regime")
        regime_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        
        self.regime_label = ttk.Label(regime_frame, text="Current Regime: ")
        self.regime_label.grid(row=0, column=0, padx=5, pady=2)
        self.regime_value = ttk.Label(regime_frame, text="Unknown")
        self.regime_value.grid(row=0, column=1, padx=5, pady=2)
        
        return regime_frame

    def create_journal_display(self):
        journal_frame = ttk.LabelFrame(self.dashboard_frame, text="Trade Journal")
        journal_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        
        # Create Treeview for trade history
        columns = ('Time', 'Market', 'Type', 'Entry', 'Exit', 'PnL', 'Risk/Reward')
        self.trade_tree = ttk.Treeview(journal_frame, columns=columns, show='headings')
        
        # Set column headings
        for col in columns:
            self.trade_tree.heading(col, text=col)
            self.trade_tree.column(col, width=100)
        
        self.trade_tree.grid(row=0, column=0, sticky='nsew')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(journal_frame, orient='vertical', command=self.trade_tree.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.trade_tree.configure(yscrollcommand=scrollbar.set)
        
        return journal_frame

    def create_risk_panel(self):
        risk_frame = ttk.LabelFrame(self.dashboard_frame, text="Risk Metrics")
        risk_frame.grid(row=1, column=2, padx=5, pady=5, sticky='nsew')
        
        # Position Size
        self.position_size_label = ttk.Label(risk_frame, text="Position Size: ")
        self.position_size_label.grid(row=0, column=0, padx=5, pady=2)
        self.position_size_value = ttk.Label(risk_frame, text="0")
        self.position_size_value.grid(row=0, column=1, padx=5, pady=2)
        
        # Risk Level
        self.risk_level_label = ttk.Label(risk_frame, text="Risk Level: ")
        self.risk_level_label.grid(row=1, column=0, padx=5, pady=2)
        self.risk_level_value = ttk.Label(risk_frame, text="Low")
        self.risk_level_value.grid(row=1, column=1, padx=5, pady=2)
        
        return risk_frame

    def update_dashboard(self):
        metrics = self.dashboard_metrics['trade_performance']
        self.win_rate_value.config(text=f"{metrics['win_rate']:.2f}%")
        self.profit_factor_value.config(text=f"{metrics['profit_factor']:.2f}")
        self.regime_value.config(text=self.dashboard_metrics['market_analysis']['regime'])

    def update_journal(self):
        # Clear existing entries
        for item in self.trade_tree.get_children():
            self.trade_tree.delete(item)
        
        # Add new entries
        trades = self.sql_operations('fetch', self.db_signals, 'Signals')
        for trade in trades[-100:]:  # Show last 100 trades
            self.trade_tree.insert('', 'end', values=(
                trade['timestamp'],
                trade['market'],
                trade['signal_type'],
                trade['price'],
                trade.get('exit_price', ''),
                trade.get('pnl', ''),
                trade.get('risk_reward', '')
            ))

    def update_risk_metrics(self):
        risk_metrics = self.dashboard_metrics['risk_metrics']
        self.position_size_value.config(text=f"{risk_metrics['position_exposure']:.2f}")
        self.risk_level_value.config(text=risk_metrics['value_at_risk'])
    def create_visualization_components(self):
        viz_frame = ttk.LabelFrame(self.dashboard_frame, text="Market Visualization")
        viz_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky='nsew')
        
        # Market Heatmap
        self.create_market_heatmap(viz_frame)
        
        # Performance Charts
        self.create_performance_charts(viz_frame)
        
        return viz_frame
    def create_advanced_analysis_panel(self):
        analysis_frame = ttk.LabelFrame(self.dashboard_frame, text="Advanced Analysis")
        analysis_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky='nsew')
        
        # Market Profile Display
        self.market_profile_canvas = self.create_market_profile_display(analysis_frame)
        
        # Order Flow Analysis
        self.order_flow_display = self.create_order_flow_display(analysis_frame)
        
        # Institutional Activity
        self.institutional_activity = self.create_institutional_display(analysis_frame)
        
        return analysis_frame

    def update_advanced_analysis(self):
        if self.scanning:
            current_market = self.get_selected_market()
            
            # Update market profile
            market_profile = self.analyze_market_profile(current_market, self.choose_time.get())
            self.update_market_profile_display(market_profile)
            
            # Update order flow
            inst_orders = self.institutional_order_detection(current_market)
            self.update_institutional_display(inst_orders)
            
            # Schedule next update
            self.master.after(10000, self.update_advanced_analysis)

    def create_market_heatmap(self, parent_frame):
        heatmap_frame = ttk.Frame(parent_frame)
        heatmap_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        
        markets = self.selected_markets or self.change(self.choose_list.get())
        performance_data = {market: self.calculate_market_performance(market) for market in markets}
        
        # Create grid of labels colored by performance
        for i, (market, perf) in enumerate(performance_data.items()):
            color = self.get_performance_color(perf)
            label = tk.Label(heatmap_frame, text=f"{market}\n{perf:.2f}%",
                            bg=color, width=15, height=2)
            label.grid(row=i//5, column=i%5, padx=1, pady=1)

    def create_performance_charts(self, parent_frame):
        chart_frame = ttk.Frame(parent_frame)
        chart_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        
        # Win Rate Chart
        self.create_win_rate_chart(chart_frame)
        
        # PnL Chart
        self.create_pnl_chart(chart_frame)

    def get_performance_color(self, performance):
        if performance > 5:
            return '#90EE90'  # Light green
        elif performance > 0:
            return '#98FB98'  # Pale green
        elif performance > -5:
            return '#FFB6C1'  # Light red
        else:
            return '#CD5C5C'  # Indian red

    def calculate_market_performance(self, market):
        trades = self.sql_operations('fetch', self.db_signals, 'Signals', market=market)
        if not trades:
            return 0.0
        
        pnl_values = [trade.get('pnl', 0) for trade in trades]
        return sum(pnl_values) / len(pnl_values) * 100

    def schedule_updates(self):
        if self.scanning:
            self.update_dashboard()
            self.update_journal()
            self.update_risk_metrics()
            self.master.after(5000, self.schedule_updates)  # Update every 5 seconds

    def start_scanning(self):
        if not self.scanning:
            # Initialize exchange connection
            self.binance = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            self.scanning = True
            self.status_label.config(text="Scanner Status: Running", fg='green')
            
            # Start scanning thread
            threading.Thread(target=self.scan_for_signals, daemon=True).start()
            
            # Notify start
            self.send_telegram_update("🚀 Scanner Started - Monitoring Markets")



    def stop_scanning(self):
        self.scanning = False
        self.status_label.config(text="Scanner Status: Stopped", fg='red')
        self.send_telegram_update("⏹️ Market Scanner Stopped")

    def user_choicelist(self):
        # Set up exchange with proper API configuration
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
        
        # Create exchange instance with public API access
        self.binance = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # Fetch market data using public endpoints
        def tickers(self):
            markets = self.binance.load_markets()
            return [market for market in markets.keys() if market.endswith('/USDT')]
        
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

  


    def analyze_market(self, market, timeframe):
        try:
            df = self.fetch_market_data(market, timeframe)
            if df is not None and not df.empty:
                # Calculate technical indicators
                df['rsi'] = talib.RSI(df['close'])
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
                df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
                df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
                
                # Volume analysis
                volume_profile = self.analyze_volume_profile(df)
                
                # Generate signals
                signals = self.generate_signals(df)
                
                # Add volume context to signals
                if volume_profile and signals:
                    for signal in signals:
                        signal['volume_context'] = volume_profile['volume_trend']
                        signal['vwap'] = volume_profile['vwap']
                
                if signals:
                    for signal in signals:
                        # Enhanced message with more context
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
                        
                        # Store enhanced signal data
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