from typing import List, Union

import numpy as np
import pandas as pd

from simulator import MdUpdate, OwnTrade, update_best_positions


def get_metrics(updates_list:List[ Union[MdUpdate, OwnTrade] ], fee=0) -> pd.DataFrame:
    '''
        This function calculates PnL from list of updates
    '''

    #current position in btc and usd
    btc_pos, usd_pos, volume = 0.0, 0.0, 0.0
    
    N = len(updates_list)
    btc_pos_arr = np.zeros((N, ))
    usd_pos_arr = np.zeros((N, ))
    mid_price_arr = np.zeros((N, ))
    trade_bid_prices = np.zeros((N, ))
    trade_ask_prices = np.zeros((N, ))
    # positions = np.zeros((N, ))
    volumes = np.zeros((N, ))

    #current best_bid and best_ask
    best_bid:float = -np.inf
    best_ask:float = np.inf
    
    for i, update in enumerate(updates_list):
        
        if isinstance(update, MdUpdate):
            best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
        #mid price
        #i use it to calculate current portfolio value
        mid_price = 0.5 * ( best_ask + best_bid )
        
        if isinstance(update, OwnTrade):
            trade = update    
            #update positions
            if trade.side == 'BID':
                btc_pos += trade.size
                usd_pos -= trade.price * trade.size
                trade_bid_prices[i] = trade.price
            elif trade.side == 'ASK':
                btc_pos -= trade.size
                usd_pos += trade.price * trade.size
                trade_ask_prices[i] = trade.price
            usd_pos -= fee * trade.price * trade.size
            volume += trade.size 
            # positions[i] = trade.size
        #current portfolio value
        
        btc_pos_arr[i] = btc_pos
        usd_pos_arr[i] = usd_pos
        mid_price_arr[i] = mid_price
        volumes[i] = volume
    
    worth_arr = btc_pos_arr * mid_price_arr + usd_pos_arr
    receive_ts = [update.receive_ts for update in updates_list]
    exchange_ts = [update.exchange_ts for update in updates_list]
    
    df = pd.DataFrame({"exchange_ts": exchange_ts, "receive_ts":receive_ts, "total": worth_arr, "BTC": btc_pos_arr, "volume": volumes, 
                       "USD": usd_pos_arr, "mid_price": mid_price_arr, "trade_bid_price": trade_bid_prices, "trade_ask_price": trade_ask_prices})
    #df = df.groupby('receive_ts').agg(lambda x: x.iloc[-1]).reset_index()    
    return df


def trade_to_dataframe(trades_list:List[OwnTrade]) -> pd.DataFrame:
    exchange_ts = [ trade.exchange_ts for trade in trades_list ]
    receive_ts = [ trade.receive_ts for trade in trades_list ]
    
    size = [ trade.size for trade in trades_list ]
    price = [ trade.price for trade in trades_list ]
    side  = [trade.side for trade in trades_list ]
    
    dct = {
        "exchange_ts" : exchange_ts,
        "receive_ts"  : receive_ts,
         "size" : size,
        "price" : price,
        "side"  : side
    }

    df = pd.DataFrame(dct).groupby('receive_ts').agg(lambda x: x.iloc[-1]).reset_index()    
    return df


def md_to_dataframe(md_list: List[MdUpdate]) -> pd.DataFrame:
    
    best_bid = -np.inf
    best_ask = np.inf
    best_bids = []
    mid_prices = []
    best_asks = []
    for md in md_list:
        best_bid, best_ask = update_best_positions(best_bid, best_ask, md)
        
        best_bids.append(best_bid)
        best_asks.append(best_ask)
        mid_prices.append((best_bid + best_ask) / 2)
        
    exchange_ts = [ md.exchange_ts for md in md_list ]
    receive_ts = [ md.receive_ts for md in md_list ]
    dct = {
        "exchange_ts" : exchange_ts,
        "receive_ts"  :receive_ts,
        "bid_price" : best_bids,
        "ask_price" : best_asks,
        'mid_price': mid_prices
    }
    
    df = pd.DataFrame(dct).groupby('receive_ts').agg(lambda x: x.iloc[-1]).reset_index()    
    return df