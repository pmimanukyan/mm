import sys

import math
import numpy as np
from collections import OrderedDict
from typing import List, Optional, Tuple, Union, Dict
from simulator import \
    MdUpdate, Order, OwnTrade, Sim, update_best_positions


class MyStrategy:
    def __init__(self, sim: Sim, b: float, eta: float, gamma: float, k: float, sigma: float,
                 terminal_time: bool, adjust_delay: int, order_size: float,
                 min_order_size: float, precision: int):
        self.b = b
        self.eta = eta

        self.sim = sim
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.terminal_time = terminal_time
        self.adjust_delay = adjust_delay
        self.order_size = order_size
        self.min_order_size = min_order_size
        self.precision = precision

        self.md_list: List[MdUpdate] = []
        self.trades_list: List[OwnTrade] = []
        self.updates_list = []
        self.all_orders = []
        
        self.ongoing_orders: OrderedDict[int, Order] = OrderedDict()
        self.best_bid = -math.inf
        self.best_ask = math.inf
        self.cur_time = 0
        self.T_minus_t = 1

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
        self.cur_pos = 0
        last_readjust = 0
        t_min = self.sim.md_queue[0].receive_ts
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

                central_price = self.get_central_price()
                if central_price is None:
                    break

                spread = 2 / self.gamma * math.log(1 + self.gamma / self.k) + 2 * self.eta + self.gamma * self.sigma**2 * self.T_minus_t
                price_bid = round(central_price - spread / 2, self.precision)
                price_ask = round(central_price + spread / 2, self.precision)
                # print(f"price_bid={price_bid}, mid_price={((price_bid + price_ask) / 2)}, price_ask={price_ask}")
                self.place_order(self.cur_time, self.order_size, 'BID', price_bid)
                self.place_order(self.cur_time, self.order_size, 'ASK', price_ask)

        return self.trades_list, self.md_list, self.updates_list, self.all_orders

    def get_central_price(self):
        """Calculates price level around which we place our maker orders"""
        midprice = (self.best_bid + self.best_ask) / 2      
        indiff_price = midprice + self.b * self.T_minus_t - self.cur_pos / self.min_order_size * (2 * self.eta + self.gamma * self.sigma**2 * self.T_minus_t)
        return indiff_price

    def place_order(self, ts: float, size: float, side: str, price: float):
        order = self.sim.place_order(ts, size, side, price)
        self.ongoing_orders[order.order_id] = order
        self.all_orders.append(order)
        
