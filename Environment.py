import pandas as pd # type: ignore
import numpy as np
import time
import matplotlib.pyplot as plt # type: ignore

class Environment:
    def __init__(self, df, window_size):
        self.step = 0
        self.window_size = window_size

        self.longs = 0
        self.shorts = 0
        self.positions = []

        self.realized_pnl = 0
        self.unrealized_pnl = 0

        self.realized_pnl_history = []
        self.unrealized_pnl_history = []

        self.trade_profit_history = []

        self.buy_signals = []
        self.sell_signals = []
        self.exit_signals = []

        self.first_rendering = False

        #########################################################################################################################################

        self.data = df.copy()

        self.data['Unrealized'] = pd.Series(dtype='float')
        self.data['Realized'] = pd.Series(dtype='float')

        self.data['longs'] = pd.Series(dtype='int')
        self.data['shorts'] = pd.Series(dtype='int')

        columns_to_fill = ['Unrealized', 'Realized', 'longs', 'shorts']
        self.data[columns_to_fill] = self.data[columns_to_fill].fillna(0)

        #########################################################################################################################################

        self.fig_price, self.ax_price = plt.subplots()

        self.ax_price.set_title('Price')
        self.ax_price.set_xlabel('Time')
        self.ax_price.set_ylabel('Price', color='blue')

        self.ax_pnl = self.ax_price.twinx()

        self.line_unrealized_pnl, = self.ax_pnl.plot([], [], color='blue', label='Unrealized PnL')
        self.line_realized_pnl, = self.ax_pnl.plot([], [], color='green', label='Realized PnL')
        self.line_price, = self.ax_price.plot([], [], color='black', label='Close Price')

        # Initialize scatter plots for buy and sell markers
        self.buy_scatter = self.ax_price.scatter([], [], marker='^', color='green', label='Buy', s=50)
        self.sell_scatter = self.ax_price.scatter([], [], marker='v', color='red', label='Sell', s=50)
        self.exit_scatter = self.ax_price.scatter([], [], marker='x', color='black', label='exit', s=50)


        self.ax_price.grid(True)

        #########################################################################################################################################

    class Position:
        def __init__(self, type, entry_price, size=0.3):
            self.type = type
            self.entry_price = entry_price
            self.size = size
            self.profit = 0

        def pnl(self, current_price):
            if self.type == 0:
                self.profit = (current_price - self.entry_price) * self.size
            elif self.type == 1:
                self.profit = (self.entry_price - current_price) * self.size

            return self.profit

    def forward(self, action):
        self.step += 1
        state = self.get_state()
        self.current_price = self.data['Close'].iloc[self.step]

        profit, executed = self.execute(action)
        done = self.step + self.window_size >= len(self.data)

        self.realized_pnl_history.append(self.realized_pnl)
        self.trade_profit_history.append(profit)

        self.unrealized_pnl = (self.pnl())
        self.unrealized_pnl_history.append(self.unrealized_pnl)

        self.realized_pnl_history.append(self.realized_pnl)

        self.data['Unrealized'] = self.unrealized_pnl
        self.data['Realized'] = self.realized_pnl
        self.data['longs'] = self.longs
        self.data['shorts'] = self.shorts

        self.render()

        return state, profit, executed, done

    def execute(self, action):
        current_time = self.data.index[self.step]
        profit = 0
        executed = 0

        if action == 0: # Nothing
            executed = 0
        
        elif action == 1 and self.longs <= 10 and self.shorts <= 0: # Long Entry
            self.long_entry(current_time)
            executed = 1

        elif action == 2 and self.shorts <= 10 and self.longs <= 0: # Short Entry
            self.short_entry(current_time)
            executed = 1

        elif action == 3: # Exit
            profit = self.exit(current_time)
            executed = 2

        return profit, executed

    def render(self):
        if self.first_rendering:
            self.first_rendering = False
            plt.ion()

        if self.step > 0:
            done = False
            while not done:
                try:
                    self.line_unrealized_pnl.set_data(self.data.index[:self.step], self.unrealized_pnl_history[:self.step])
                    self.line_realized_pnl.set_data(self.data.index[:self.step], self.realized_pnl_history[:self.step])
                    self.line_price.set_data(self.data.index[:self.step], self.data['Close'].iloc[:self.step])

                    # Plot buy and sell signals
                    buy_indices = [i for i in range(len(self.buy_signals)) if self.buy_signals[i][1] > 0]
                    sell_indices = [i for i in range(len(self.sell_signals)) if self.sell_signals[i][1] > 0]
                    exit_indices = [i for i in range(len(self.exit_signals)) if self.exit_signals[i][1] > 0]

                    # Extract timestamps and y-values for buy/sell signals
                    buy_times = [self.buy_signals[i][0] for i in buy_indices]
                    buy_prices = [self.buy_signals[i][1] for i in buy_indices]

                    sell_times = [self.sell_signals[i][0] for i in sell_indices]
                    sell_prices = [self.sell_signals[i][1] for i in sell_indices]

                    exit_times = [self.exit_signals[i][0] for i in exit_indices]
                    exit_prices = [self.exit_signals[i][1] for i in exit_indices]

                    # Convert to numpy arrays if necessary
                    buy_times = np.asarray(buy_times, dtype=float)
                    buy_prices = np.asarray(buy_prices, dtype=float)

                    sell_times = np.asarray(sell_times, dtype=float)
                    sell_prices = np.asarray(sell_prices, dtype=float)

                    exit_times = np.asarray(exit_times, dtype=float)
                    exit_prices = np.asarray(exit_prices, dtype=float)

                    # Set offsets for buy and sell signals (timestamps and corresponding price)
                    if len(buy_times) > 0:
                        self.buy_scatter.set_offsets(np.column_stack((buy_times, buy_prices)))

                    if len(sell_times) > 0:
                        self.sell_scatter.set_offsets(np.column_stack((sell_times, sell_prices)))

                    if len(exit_times) > 0:
                        self.exit_scatter.set_offsets(np.column_stack((exit_times, exit_prices)))


                    self.ax_price.relim()
                    self.ax_price.autoscale_view()
                    self.ax_pnl.relim()
                    self.ax_pnl.autoscale_view()


                    plt.draw()
                    plt.pause(0.005)
                    done = True

                except ValueError:
                    continue

    #########################################################################################################################################
    
    def get_state(self):
        start = self.step
        end = start + self.window_size
        features = self.data.to_numpy()[start:end]
        return features.astype(np.float32)

    def reset(self):
        self.step = 0

        self.longs = 0
        self.shorts = 0
        self.positions = []

        self.realized_pnl = 0
        self.unrealized_pnl = 0

        self.realized_pnl_history = []
        self.unrealized_pnl_history = []

        self.trade_profit_history = []

        self.buy_signals = []
        self.sell_signals = []
        self.exit_signals = []

        self.first_rendering = False

        columns_to_fill = ['Unrealized', 'Realized', 'longs', 'shorts']
        self.data[columns_to_fill] = self.data[columns_to_fill].fillna(0)

        return self.get_state()

    def close(self):
        plt.close(self.fig_price)
        print("closing")

    def pnl(self):
        profit = 0
        for position in self.positions:
            profit += position.pnl(self.current_price)

        return profit

    #########################################################################################################################################

    def long_entry(self, time):
        self.positions.append(self.Position(0, self.current_price))
        self.buy_signals.append((time, self.current_price))
        self.longs += 1
        
    def short_entry(self, time):
        self.positions.append(self.Position(1, self.current_price))
        self.sell_signals.append((time, self.current_price))
        self.shorts += 1

    def exit(self, time):
        profit = 0

        for position in self.positions:
            profit += position.pnl(self.current_price)

        print(f'exited, profit: {profit}')

        self.exit_signals.append((time, self.current_price))
        self.realized_pnl += profit
        self.positions = []
        self.longs = 0
        self.shorts = 0

        return profit