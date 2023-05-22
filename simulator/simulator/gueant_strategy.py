import sys
import os
sys.path.append('../..')

import math
import numpy as np
from collections import OrderedDict
from typing import List, Optional, Tuple, Union, Dict
from simulator import \
    MdUpdate, Order, OwnTrade, Sim, update_best_positions

class MyStrategy:
    def __init__(self, sim: Sim, A: float, gamma: float, k: float, sigma: float,
                 terminal_time: bool, adjust_delay: int, order_size: float,
                 min_order_size: float, precision: int):
        self.A = A


        self.sim = sim
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.terminal_time = terminal_time
        self.adjust_delay = adjust_delay
        self.order_size = order_size
        self.min_order_size = min_order_size
        self.precision = precision

        self.md_list: List[MdUpdate] = []      # market data list
        self.trades_list: List[OwnTrade] = []  # executed trades list
        self.updates_list = []                 # all updates list
        self.all_orders = []                   # all orders list
        # orders that have not been executed/canceled yet
        self.ongoing_orders: OrderedDict[int, Order] = OrderedDict()
        self.best_bid = -math.inf  # current best bid
        self.best_ask = math.inf   # current best ask
        self.cur_time = 0          # current time
        self.T_minus_t = 1         # term (T-t) from (29), (30) in [Stoikov 2008]

    def run(self) -> Tuple[List[OwnTrade], List[MdUpdate],
                           List[Union[OwnTrade, MdUpdate]], List[Order]]:
        """This function runs the simulation.

            Args:
            Returns:
                trades_list(List[OwnTrade]):
                    List of our executed trades.
                md_list(List[MdUpdate]):
                    List of market data received by strategy.
                updates_list( List[Union[OwnTrade, MdUpdate]] ):
                    List of all updates received by strategy
                    (market data and information about executed trades).
                all_orders(List[Orted]):
                    List of all placed orders.
        """
        # current position size in base asset
        self.cur_pos = 0
        # timestamp when last rebalancing happened
        last_readjust = 0
        t_min = self.sim.md_queue[0].receive_ts
        # terminal time T from (29), (30) in [Stoikov 2008]
        t_max = self.sim.md_queue[-1].receive_ts

        while True:
            # get update from simulator
            self.cur_time, updates = self.sim.tick()
            if updates is None:
                break
            # save updates
            self.updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    self.best_bid, self.best_ask = update_best_positions(
                        self.best_bid, self.best_ask, update)
                    self.md_list.append(update)
                elif isinstance(update, OwnTrade):
                    own_trade = update
                    if own_trade.side == 'BID':
                        self.cur_pos += own_trade.size
                    else:
                        self.cur_pos -= own_trade.size
                    self.trades_list.append(own_trade)
                    # delete executed trades from the dict
                    if own_trade.order_id in self.ongoing_orders.keys():
                        self.ongoing_orders.pop(own_trade.order_id)
                else:
                    assert False, 'invalid type of update!'

            if self.cur_time - last_readjust > self.adjust_delay:
                last_readjust = self.cur_time
                # cancel all orders
                while self.ongoing_orders:
                    order_id, _ = self.ongoing_orders.popitem(last=False)
                    self.sim.cancel_order(self.cur_time, order_id)

                if self.terminal_time:
                    t = (self.cur_time - t_min) / (t_max - t_min)  # normalize
                    self.T_minus_t = 1 - t
                else:
                    self.T_minus_t = 1

                # page 11 Gueant 2012
                mid_price = (self.best_bid + self.best_ask) / 2
                if mid_price is None:
                    break
                ask_delta = 1 / self.gamma * math.log(1 + self.gamma / self.k) + ((2 * self.cur_pos + 1) / 2) * math.sqrt((self.sigma * self.gamma) / (2 * self.k * self.A) * (1 + self.gamma / self.k)**(1 + self.k / self.gamma))
                bid_delta = 1 / self.gamma * math.log(1 + self.gamma / self.k) - ((2 * self.cur_pos - 1) / 2) * math.sqrt((self.sigma * self.gamma) / (2 * self.k * self.A) * (1 + self.gamma / self.k)**(1 + self.k / self.gamma))
                
                price_bid = round(mid_price - bid_delta, self.precision)
                price_ask = round(mid_price + ask_delta, self.precision)
                # print(f"bid_price:{price_bid}, mid_price:{mid_price}, ask_price:{price_ask}")
                self.place_order(self.cur_time, self.order_size, 'BID', price_bid)
                self.place_order(self.cur_time, self.order_size, 'ASK', price_ask)

        return self.trades_list, self.md_list, self.updates_list, self.all_orders

    def get_central_price(self):
        """Calculates price level around which we place our maker orders"""
        midprice = (self.best_bid + self.best_ask) / 2
        # indifference price, (29) from [Stoikov 2008]
        indiff_price = midprice - self.cur_pos / self.min_order_size \
                       * self.gamma * self.sigma**2 * self.T_minus_t
        return indiff_price

    def place_order(self, ts: float, size: float, side: str, price: float):
        order = self.sim.place_order(ts, size, side, price)
        self.ongoing_orders[order.order_id] = order
        self.all_orders.append(order)