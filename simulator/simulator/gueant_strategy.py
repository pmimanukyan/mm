# import sys
# import os
# sys.path.append('../..')

# import math
# import numpy as np
# from collections import OrderedDict
# from typing import List, Optional, Tuple, Union, Dict
# from simulator import \
#     MdUpdate, Order, OwnTrade, Sim, update_best_positions

# """
# References:
#      [Stoikov 2008] Avellaneda, M., & Stoikov, S. (2008). High-frequency
#      trading in a limit order book. Quantitative Finance, 8(3), 217-224.
# """


# class MyStrategy:
#     """Strategy from [Stoikov 2008]"""

#     def __init__(self, sim: Sim, A: float, gamma: float, k: float, sigma: float,
#                  terminal_time: bool, adjust_delay: int, order_size: float,
#                  min_order_size: float, precision: int):
#         """
#         Args:
#             sim:
#                 Exchange simulator.
#             gamma:
#                 Parameter γ from (29), (30) in [Stoikov 2008]. Assumed to be
#                 non-negative. Small values correspond to more risk neutral
#                 strategy, larger values correspond to more risk averse strategy.
#             k:
#                 Parameter k from (30) in [Stoikov 2008]. A statistic that is
#                 calculated from the market data.
#             sigma:
#                 Parameter σ from (29) in [Stoikov 2008]. A standard deviation
#                 of increments of the Wiener process that is assumed to be the
#                 model for the asset price.
#             terminal_time:
#                 Whether the terminal time T exists or not. If `True`, terminal
#                 time is the timestamp of the last element in simulator market
#                 data queue. If `False`, the term (T-t) is set to 1.
#             adjust_delay:
#                 Delay (in nanoseconds) between readjusting the orders.
#             order_size:
#                 Size of limit orders placed by the strategy.
#             min_order_size:
#                 Minimum order size for the base asset allowed by exchange. E.g.
#                 0.001 BTC for Binance BTC/USDT Perpetual Futures as of 2022-11-29.
#             precision:
#                 Precision of the price - a number of decimal places. E.g. 2 for
#                 Binance BTC/USDT Perpetual Futures as of 2022-11-29.
#         """
#         self.A = A


#         self.sim = sim
#         self.gamma = gamma
#         self.k = k
#         self.sigma = sigma
#         self.terminal_time = terminal_time
#         self.adjust_delay = adjust_delay
#         self.order_size = order_size
#         self.min_order_size = min_order_size
#         self.precision = precision

#         self.md_list: List[MdUpdate] = []      # market data list
#         self.trades_list: List[OwnTrade] = []  # executed trades list
#         self.updates_list = []                 # all updates list
#         self.all_orders = []                   # all orders list
#         # orders that have not been executed/canceled yet
#         self.ongoing_orders: OrderedDict[int, Order] = OrderedDict()
#         self.best_bid = -math.inf  # current best bid
#         self.best_ask = math.inf   # current best ask
#         self.cur_time = 0          # current time
#         self.T_minus_t = 1         # term (T-t) from (29), (30) in [Stoikov 2008]

#     def run(self) -> Tuple[List[OwnTrade], List[MdUpdate],
#                            List[Union[OwnTrade, MdUpdate]], List[Order]]:
#         """This function runs the simulation.

#             Args:
#             Returns:
#                 trades_list(List[OwnTrade]):
#                     List of our executed trades.
#                 md_list(List[MdUpdate]):
#                     List of market data received by strategy.
#                 updates_list( List[Union[OwnTrade, MdUpdate]] ):
#                     List of all updates received by strategy
#                     (market data and information about executed trades).
#                 all_orders(List[Orted]):
#                     List of all placed orders.
#         """
#         # current position size in base asset
#         self.cur_pos = 0
#         # timestamp when last rebalancing happened
#         last_readjust = 0
#         t_min = self.sim.md_queue[0].receive_ts
#         # terminal time T from (29), (30) in [Stoikov 2008]
#         t_max = self.sim.md_queue[-1].receive_ts

#         while True:
#             # get update from simulator
#             self.cur_time, updates = self.sim.tick()
#             if updates is None:
#                 break
#             # save updates
#             self.updates_list += updates
#             for update in updates:
#                 # update best position
#                 if isinstance(update, MdUpdate):
#                     self.best_bid, self.best_ask = update_best_positions(
#                         self.best_bid, self.best_ask, update)
#                     self.md_list.append(update)
#                 elif isinstance(update, OwnTrade):
#                     own_trade = update
#                     if own_trade.side == 'BID':
#                         self.cur_pos += own_trade.size
#                     else:
#                         self.cur_pos -= own_trade.size
#                     self.trades_list.append(own_trade)
#                     # delete executed trades from the dict
#                     if own_trade.order_id in self.ongoing_orders.keys():
#                         self.ongoing_orders.pop(own_trade.order_id)
#                 else:
#                     assert False, 'invalid type of update!'

#             if self.cur_time - last_readjust > self.adjust_delay:
#                 last_readjust = self.cur_time
#                 # cancel all orders
#                 while self.ongoing_orders:
#                     order_id, _ = self.ongoing_orders.popitem(last=False)
#                     self.sim.cancel_order(self.cur_time, order_id)

#                 if self.terminal_time:
#                     t = (self.cur_time - t_min) / (t_max - t_min)  # normalize
#                     self.T_minus_t = 1 - t
#                 else:
#                     self.T_minus_t = 1

#                 # page 11 Gueant 2012
#                 mid_price = (self.best_bid + self.best_ask) / 2
#                 if mid_price is None:
#                     break
#                 ask_delta = 1 / self.gamma * math.log(1 + self.gamma / self.k) + ((2 * self.cur_pos + 1) / 2) * math.sqrt((self.sigma * self.gamma) / (2 * self.k * self.A) * (1 + self.gamma / self.k)**(1 + self.k / self.gamma))
#                 bid_delta = 1 / self.gamma * math.log(1 + self.gamma / self.k) - ((2 * self.cur_pos - 1) / 2) * math.sqrt((self.sigma * self.gamma) / (2 * self.k * self.A) * (1 + self.gamma / self.k)**(1 + self.k / self.gamma))
                
#                 price_bid = round(mid_price - bid_delta, self.precision)
#                 price_ask = round(mid_price + ask_delta, self.precision)
#                 # print(f"bid_price:{price_bid}, mid_price:{mid_price}, ask_price:{price_ask}")
#                 self.place_order(self.cur_time, self.order_size, 'BID', price_bid)
#                 self.place_order(self.cur_time, self.order_size, 'ASK', price_ask)

#         return self.trades_list, self.md_list, self.updates_list, self.all_orders

#     def get_central_price(self):
#         """Calculates price level around which we place our maker orders"""
#         midprice = (self.best_bid + self.best_ask) / 2
#         # indifference price, (29) from [Stoikov 2008]
#         indiff_price = midprice - self.cur_pos / self.min_order_size \
#                        * self.gamma * self.sigma**2 * self.T_minus_t
#         return indiff_price

#     def place_order(self, ts: float, size: float, side: str, price: float):
#         order = self.sim.place_order(ts, size, side, price)
#         self.ongoing_orders[order.order_id] = order
#         self.all_orders.append(order)

from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator import MdUpdate, Order, OwnTrade, Sim, update_best_positions

class GueantStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, trade_size: float, position_limit: float, delay: float, hold_time:Optional[float] = None, risk_aversion:Optional[float] = 0.5, difference:Optional[float] = 0, A:Optional[float] = 1) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time
        self.order_size = trade_size
        self.last_mid_prices = []
        self.correction = difference
        self.gamma = risk_aversion
        self.Q = position_limit
        self.A = A
        self.asset_position = 0
        self.current_bid_order_id = None
        self.current_ask_order_id = None
        self.previous_bid_order_id = None
        self.previous_ask_order_id = None
        self.trades_dict = {'place_ts' :[], 'exchange_ts': [], 'receive_ts': [], 'trade_id': [],'order_id': [],'side': [], 'size': [], 'price': [],'execute':[], 'mid_price':[]}  
    
    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        best_bid = -np.inf
        best_ask = np.inf

        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    mid_price = (best_bid + best_ask)/2
                    if len(self.last_mid_prices) < 500:
                        self.last_mid_prices.append(mid_price)
                    else:
                        self.last_mid_prices.append(mid_price)
                        self.last_mid_prices.pop(0)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    self.trades_dict['place_ts'].append(update.place_ts)
                    self.trades_dict['exchange_ts'].append(update.exchange_ts)
                    self.trades_dict['receive_ts'].append(update.receive_ts)
                    self.trades_dict['trade_id'].append(update.trade_id)
                    self.trades_dict['order_id'].append(update.order_id)
                    self.trades_dict['side'].append(update.side)
                    self.trades_dict['size'].append(update.size)
                    self.trades_dict['price'].append(update.price)
                    self.trades_dict['execute'].append(update.execute)
                    self.trades_dict['mid_price'].append(mid_price)
                    if update.side == "ASK":
                        self.asset_position -= update.size
                    elif update.side == "BID":
                        self.asset_position += update.size
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                #place order
                '''
                s      : current mid_price
                q      : current position in asset
                sigma  : parameter in the Geometric Brownian Motion equation [dS_t = sigma dw_t]
                gamma  : risk-aversion parameter of the optimizing agents (across the economy)
                xi     : often referred to as the gamma parameter, but has slightly different meaning - the magnitude towards exponential utility function over linear utility function
                delta  : or equvalently the size of limit orders
                K      : higher K means that market order volumes have higher impact on best price changes
                alpha  : higher alpha means higher probability fo large market orders
                A      : scaling parameter in the density function of market order size 
                Q      : limit of position, the market maker stops providing the limit orders that could make the asset_position violate the limit
                
                '''
                xi = self.gamma - self.correction
                if len(self.last_mid_prices)==500:
                    sigma = np.std(self.last_mid_prices)## per update --> need to scale it to the "per second" terminology
                else:
                    sigma = 1
                delta_t = 0.032 ## there is approximately 0.032 seconds in between the orderbook uprates (nanoseconds / 1e9 = seconds)
                sigma = sigma*np.sqrt(1/delta_t)
                k = 1.5
                q = self.asset_position
                delta_ = self.order_size
                ## mid_price was defined previously
                if xi == 0:
                    delta_ask = 1/k - (2*q-delta_)/2*np.sqrt( self.gamma*(sigma**2)*np.exp(1)/2/self.A/delta_/k )
                    delta_bid = 1/k + (2*q+delta_)/2*np.sqrt( self.gamma*(sigma**2)*np.exp(1)/2/self.A/delta_/k )
                else:
                    delta_ask = 1/xi/delta_ * np.log(1 + xi*delta_/k) - (2*q - delta_)/2*np.sqrt( self.gamma*(sigma**2)/2/self.A/delta_/k * (1+xi*delta_/k)**(1+k/xi/delta_) )
                    delta_bid = 1/xi/delta_ * np.log(1 + xi*delta_/k) + (2*q + delta_)/2*np.sqrt( self.gamma*(sigma**2)/2/self.A/delta_/k * (1+xi*delta_/k)**(1+k/xi/delta_) )
                
                bid_price = np.round(mid_price - delta_bid, 1)
                ask_price = np.round(mid_price + delta_ask, 1)
                if (self.asset_position < self.Q):
                    bid_order = sim.place_order( receive_ts, self.order_size, 'BID', bid_price)
                    ongoing_orders[bid_order.order_id] = bid_order
                    self.previous_bid_order_id = self.current_bid_order_id
                    self.current_bid_order_id = bid_order.order_id
                    all_orders.append(bid_order)
                    
                if (self.asset_position > -self.Q):
                    ask_order = sim.place_order( receive_ts, self.order_size, 'ASK', ask_price)
                    ongoing_orders[ask_order.order_id] = ask_order
                    self.previous_ask_order_id = self.current_ask_order_id
                    self.current_ask_order_id = ask_order.order_id
                    all_orders.append(ask_order)
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            if self.previous_bid_order_id in ongoing_orders.keys():
                sim.cancel_order( receive_ts, self.previous_bid_order_id )
                to_cancel.append(self.previous_bid_order_id)
            if self.previous_ask_order_id in ongoing_orders.keys():
                sim.cancel_order( receive_ts, self.previous_ask_order_id )
                to_cancel.append(self.previous_ask_order_id)
            for ID in to_cancel:
                try:
                    ongoing_orders.pop(ID)
                except KeyError:
                    print('tried to cancel ', ID, '. Something went wrong.')
            
                
        return trades_list, md_list, updates_list, all_orders
