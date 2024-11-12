import tkinter as tk
from tkinter import ttk
from base import BaseScanner
import ccxt
import sqlite3
from datetime import datetime
import time
import csv
import requests
from market_app import MarketApp
import talib
import pandas as pd
from tkinter import messagebox 
from market_scanner import MarketScanner
from vsa_analyzer import VSAAnalyzer

class ScannerGUI(MarketScanner):
    def __init__(self, master=None):
        super().__init__(master)
        self.db_connect = "connection.db"
        self.db_signals = "Signals.db"
        
        self.sql_operations('create', self.db_connect, 'userinfo')
        self.sql_operations('create', self.db_signals, 'Signals')
        
        self.tel_id = None
        self.bot_token = None 
        self.scanning = False
        self.filtered_list = []
        self.selected_markets = []
        self.timeframesdic = {
            '1s': 0, '1m': 1, '5m': 2, '15m': 3,
            '30m': 4, '1h': 5, '4h': 6, '1d': 7
        }
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

                if self.con('connection.db'):  # Added file_name argument here
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

                if self.con('connection.db'):  # Added file_name argument here
                    self.api_key = str(private_key)
                    self.secret_key = str(secret_key)
                    self.phrase = phraseword
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
        list_manu_ex = ['Binance', 'CoinEx', 'okx', 'BingX']
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

    def start_scanning(self):
        if not self.scanning:
            if self.create_binance_object():
                self.scanning = True
                self.status_label.config(text="Scanner Status: Running", fg='green')
                self.run(self.scan_for_signals)
                self.send_telegram_update("🚀 Market Scanner Started\nMonitoring markets for signals...")
            else:
                self.status_label.config(text="Scanner Status: Error", fg='red')

    def stop_scanning(self):
        self.scanning = False
        self.status_label.config(text="Scanner Status: Stopped", fg='red')
        self.send_telegram_update("⏹️ Market Scanner Stopped")
    def user_choicelist(self):
        self.binance = self.create_binance_object()
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

    def scan_for_signals(self):
        while self.scanning:
            try:
                if self.user_choice.get() == 0:
                    markets = self.change(self.choose_list.get())
                else:
                    markets = self.selected_markets

                for market in markets:
                    if not self.scanning:
                        break
                    
                    timeframe = self.choose_time.get()
                    self.analyze_market(market, timeframe)
                    time.sleep(0.5)  # Rate limiting

            except Exception as e:
                self.logger.error(f"Scanning error: {e}")
                time.sleep(5)

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

    def generate_signals(self, df):
        signals = []
        try:
            # Calculate technical indicators
            df['rsi'] = talib.RSI(df['close'])
            df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
            df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
            
            # Check for bullish conditions
            if df['rsi'].iloc[-1] < 30:  # Oversold
                signals.append({
                    'type': 'RSI_OVERSOLD',
                    'price': df['close'].iloc[-1]
                })
                
            if df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1] and \
            df['ema_20'].iloc[-2] <= df['ema_50'].iloc[-2]:  # Golden Cross
                signals.append({
                    'type': 'EMA_CROSS',
                    'price': df['close'].iloc[-1]
                })
                
            return signals
        except Exception as e:
            print(f"Error generating signals: {e}")
            return []

    def process_signals(self, market, timeframe, signals):
        for signal in signals:
            # Store signal in database
            self.sql_operations('insert', self.db_signals, 'Signals', 
                            market=market,
                            timeframe=timeframe,
                            signal_type=signal['type'],
                            price=signal['price'],
                            timestamp=str(datetime.now()))
            
            # Send notification
            message = f"Signal: {signal['type']}\nMarket: {market}\nTimeframe: {timeframe}\nPrice: {signal['price']}"
            self.send_telegram_update(message)
            

    def analyze_market(self, market, timeframe):
        try:
            df = self.fetch_market_data(market, timeframe)
            if df is not None and not df.empty:
                # Regular technical analysis
                signals = self.generate_signals(df)
                
                # VSA analysis
                vsa_analyzer = VSAAnalyzer()
                vsa_df = vsa_analyzer.analyze_candles(df)
                
                # Support volume analysis
                support_signals = vsa_analyzer.analyze_support_volume(vsa_df)
                
                if support_signals:
                    for signal in support_signals:
                        message = (
                            f"🔔 High Volume Support Rejection!\n"
                            f"Market: {market}\n"
                            f"Timeframe: {timeframe}\n"
                            f"Price: {signal['price']}\n"
                            f"Support Level: {signal['support_level']}\n"
                            f"Volume Spike: {signal['volume']:.2f}\n"
                            f"Wick Ratio: {signal['wick_ratio']:.2%}"
                        )
                        self.send_telegram_update(message)
                        
                        fig = vsa_analyzer.plot_vsa_analysis(vsa_df, market)
                        if fig:
                            self.send_chart_to_telegram(market, timeframe, fig)
                
                if signals or (vsa_df is not None and vsa_df['VSA_Pattern'].iloc[-1] != 'Neutral'):
                    self.process_signals(market, timeframe, signals)
                    
        except Exception as e:
            self.logger.error(f"Error analyzing market {market}: {e}")


    def fetch_market_data(self, market, timeframe):
        try:
            ohlcv = self.binance.fetch_ohlcv(market, timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            self.logger.error(f"Data fetch error for {market}: {e}")
            return None

    def process_signals(self, market, timeframe, signals):
        for signal in signals:
            self.sql_operations('insert', self.db_signals, 'Signals', 
                              market=market,
                              timeframe=timeframe,
                              signal_type=signal['type'],
                              price=signal['price'],
                              timestamp=str(datetime.now()))
            
            message = f"Signal: {signal['type']}\nMarket: {market}\nTimeframe: {timeframe}\nPrice: {signal['price']}"
            self.send_telegram_update(message)
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
