import tkinter as tk
from tkinter import ttk
import csv

class MarketApp(tk.Toplevel):
    def __init__(self, master, market_names):
        super().__init__(master)
        self.master = master
        self.market_names = market_names
        self.title("Market Selection")
        self.geometry("800x600")

        # Ensure selected_markets is initialized
        if not hasattr(self.master, 'selected_markets'):
            self.master.selected_markets = []

        # Split markets into two parts for display
        midpoint = len(self.market_names) // 2
        self.markets_part1 = self.market_names[:midpoint]
        self.markets_part2 = self.market_names[midpoint:]

        # Setup GUI elements
        self.setup_widgets()

    def setup_widgets(self):
        # First market list
        self.market_label = ttk.Label(self, text="Markets 1:")
        self.market_listbox = tk.Listbox(self, selectmode='multiple', exportselection=False, height=20, width=30)
        for market in self.markets_part1:
            self.market_listbox.insert(tk.END, market)

        self.market_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.market_listbox.grid(row=1, column=0, padx=10, pady=5, sticky='nsew')

        # Second market list
        self.market_label1 = ttk.Label(self, text="Markets 2:")
        self.market_listbox1 = tk.Listbox(self, selectmode='multiple', exportselection=False, height=20, width=30)
        for market in self.markets_part2:
            self.market_listbox1.insert(tk.END, market)

        self.market_label1.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        self.market_listbox1.grid(row=1, column=1, padx=10, pady=5, sticky='nsew')

        # Search functionality
        self.search_label = ttk.Label(self, text="Search Markets:")
        self.search_label.grid(row=2, column=0, padx=10, pady=5, sticky='w')
        
        self.search_entry = ttk.Entry(self, width=30)
        self.search_entry.grid(row=2, column=1, padx=10, pady=5, sticky='w')
        self.search_entry.bind("<KeyRelease>", self.filter_markets)

        # Selected markets display
        self.selected_markets_label = ttk.Label(self, text="Selected Markets:")
        self.selected_markets_listbox = tk.Listbox(self, selectmode='multiple', exportselection=False, height=20, width=30)
        self.selected_markets_label.grid(row=0, column=2, padx=10, pady=5, sticky='w')
        self.selected_markets_listbox.grid(row=1, column=2, padx=10, pady=5, sticky='nsew')

        # Control buttons frame
        button_frame = ttk.Frame(self)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)

        # Control buttons
        self.add_button = ttk.Button(button_frame, text="Add Selected", command=self.add_selected_markets)
        self.save_button = ttk.Button(button_frame, text="Save", command=self.save_markets_to_csv)
        self.load_button = ttk.Button(button_frame, text="Load", command=self.load_markets_from_csv)
        self.delete_button = ttk.Button(button_frame, text="Delete", command=self.delete_selected_markets)
        self.submit_button = ttk.Button(button_frame, text="Submit and Close", command=self.submit_and_close)

        # Grid layout for buttons
        self.add_button.grid(row=0, column=0, padx=5)
        self.save_button.grid(row=0, column=1, padx=5)
        self.load_button.grid(row=0, column=2, padx=5)
        self.delete_button.grid(row=0, column=3, padx=5)
        self.submit_button.grid(row=0, column=4, padx=5)

        # Configure grid weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(1, weight=1)

    def submit_and_close(self):
        self.destroy()

    def update_selected_markets_listbox(self):
        self.selected_markets_listbox.delete(0, tk.END)
        for market in self.master.selected_markets:
            self.selected_markets_listbox.insert(tk.END, market)

    def filter_markets(self, event):
        search_query = self.search_entry.get().lower()
        self.filter_listbox(self.market_listbox, self.markets_part1, search_query)
        self.filter_listbox(self.market_listbox1, self.markets_part2, search_query)
    def filter_markets_by_criteria(self, markets):
        filtered_markets = []
        for market in markets:
            try:
                # Volume Check
                volume = self.get_market_volume(market)
                if volume < self.min_volume:
                    continue
                    
                # Spread Check
                spread = self.get_market_spread(market)
                if spread > self.max_spread:
                    continue
                    
                # Volatility Check
                volatility = self.get_market_volatility(market)
                if volatility < self.min_volatility:
                    continue
                    
                filtered_markets.append(market)
                
            except Exception as e:
                self.logger.error(f"Error filtering market {market}: {e}")
                
        return filtered_markets

    def filter_listbox(self, listbox, markets, search_query):
        listbox.delete(0, tk.END)
        for market in markets:
            if search_query in market.lower():
                listbox.insert(tk.END, market)

    def add_selected_markets(self):
        for listbox in (self.market_listbox, self.market_listbox1):
            selected_indices = listbox.curselection()
            for index in selected_indices:
                market = listbox.get(index)
                if market not in self.master.selected_markets:
                    self.master.selected_markets.append(market)
        
        self.market_listbox1.selection_clear(0, tk.END)
        self.market_listbox.selection_clear(0, tk.END)
        self.update_selected_markets_listbox()

    def save_markets_to_csv(self):
        with open('selected_markets.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for market in self.master.selected_markets:
                writer.writerow([market])

    def load_markets_from_csv(self):
        self.master.selected_markets.clear()
        self.selected_markets_listbox.delete(0, tk.END)
        try:
            with open('selected_markets.csv', 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    self.master.selected_markets.append(row[0])
        except FileNotFoundError:
            print("CSV file not found. Please save the markets first.")
        
        self.update_selected_markets_listbox()
    def track_market_performance(self):
        performance_data = {}
        for market in self.master.selected_markets:
            signals = self.sql_operations('fetch', self.db_signals, 'Signals', 
                                        market=market)
            success_rate = self.calculate_signal_success(signals)
            performance_data[market] = success_rate
        return performance_data

    def delete_selected_markets(self):
        selected_indices = self.selected_markets_listbox.curselection()
        for index in reversed(selected_indices):
            del self.master.selected_markets[index]
        self.update_selected_markets_listbox()

    def run_choiceApp(self):
        self.mainloop()