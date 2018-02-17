#!/bin/python3

import os
import sys
from decimal import *
getcontext().prec = 7

""" Equity Order Matching: Goldman Codesprint 2018"""

__author__      = "Satyam Mittal"
__copyright__ = "Copyright 2018"
class stocks():

    def __init__(self):
        self.size = 0
        self.stock = []
        self.time = 0
        self.buy = {}
        self.sell = {}
        self.stock_id = {}
        self.queriesQ = {}
        self.timesQ = {}
        self.ansQ = {}

    def check_ordertype(self, order_type):
        if order_type in ['M', 'L', 'I']:
            return True
        else:
            return False

    def check_side(self, side):
        if side in ['B', 'S']:
            return True
        else:
            return False

    def check_quantity(self, quantity):
        quantity = float(quantity)
        if (int(quantity) == float(quantity)) and int(quantity) > 0:
            return True
        else:
            return False

    def weirdquery(self, time):
        if self.time <= time:
            self.time = time
            return False
        else:
            return True

    def add_to_state(self):
        i = len(self.stock) - 1
        st = self.stock[i]
        st_symbol = st["symbol"]
        st_type = st["side"]
        st_id = st["id"]
        if st_type == "B":
            if st_symbol not in self.buy.keys():
                self.buy[st_symbol] = [st_id]
            else:
                self.buy[st_symbol].append(st_id)
        else:
            if st_symbol not in self.sell.keys():
                self.sell[st_symbol] = [st_id]
            else:
                self.sell[st_symbol].append(st_id)
        return

    def checkprice(self, price):
        fprice = float(price)
        if(fprice >= 0):
            return True
        else:
            return False

    def add(self, stock_id, time, symbol, type_stock, side, price, quantity, count):
        try:
            if self.weirdquery(int(time)):
                return []
            if (int(stock_id) > 0 and stock_id not in self.stock_id.keys() and self.check_ordertype(type_stock)
                    and self.check_side(side) and self.check_quantity(quantity) and symbol.isalpha() and self.checkprice(price)):
                if type_stock == 'M' and float(price) != 0:
                    raise Exception
                self.stock_id[stock_id] = len(self.stock)
                cprice = 0
                if type_stock == 'M':
                    cprice = 1
                self.stock.append({"id": stock_id, "time": int(time), "symbol": symbol, "type": type_stock,
                                   "side": side, "price": float(price), "quantity": int(quantity), "market": cprice,
                                   "per_quantity": int(quantity), "closed": False, "fifo": count})
                self.add_to_state()
                return [("{order} - Accept").format(order=stock_id)]
            else:
                return [("{order} - Reject - 303 - Invalid order details".format(order=stock_id))]
        except Exception as e:
            # print(e)
            return [("{order} - Reject - 303 - Invalid order details".format(order=stock_id))]

    def get_stock(self, arr_id):
        arr_id = str(arr_id)
        if arr_id in self.stock_id.keys() and not self.stock[self.stock_id[arr_id]]["closed"]:
            return self.stock[self.stock_id[arr_id]]
        return None

    def update(self, stock):
        stc = str(stock["id"])
        # print(self.stock_id)
        self.stock[self.stock_id[stc]] = stock
        return

    def append(self, stock_id, time, symbol, type_stock, side, price, quantity):
        if self.weirdquery(int(time)):
            return []
        stc = self.get_stock(stock_id)
        try:
            stock_id_int = int(stock_id)
            if(not self.check_side(side) or not self.check_ordertype(type_stock) or
                not self.check_quantity(quantity) or float(price) < 0 or not symbol.isalpha() or
                    not self.checkprice(price)):
                raise Exception
            if stc is not None:
                if (stc["symbol"] == symbol and stc["type"] == type_stock and
                        stc["side"] == side):
                    if stc["price"] == float(price) and stc["quantity"] == int(quantity):
                        raise Exception
                    if stc["price"] != float(price) and stc["quantity"] == int(quantity):
                        stc["time"] = int(time)
                    if stc["quantity"] < int(quantity):
                        stc["time"] = int(time)
                    stc["price"] = float(price)
                    if stc["type"] == 'M' and stc["price"] != 0:
                        raise Exception
                    total_quantity = stc["per_quantity"]
                    diff = total_quantity - int(quantity)
                    stc["quantity"] = max(0, stc["quantity"] - diff)
                    stc["per_quantity"] = int(quantity)
                    if stc["quantity"] == 0:
                        stc["closed"] = True
                    #stc["time"] = int(time)
                    self.update(stc)
                    return [("{order} - AmendAccept").format(order=stock_id)]
                else:
                    return [("{order} - AmendReject - 101 - Invalid amendment details".format(order=stock_id))]
            else:
                return [("{order} - AmendReject - 404 - Order does not exist".format(order=stock_id))]
        except Exception as e:
            return [("{order} - AmendReject - 101 - Invalid amendment details".format(order=stock_id))]

    def return_stocks_list(self, lst, tag):
        result = []
        for stc_id in lst:
            if self.get_stock(stc_id) != None:
                result.append(self.get_stock(stc_id))
        from operator import itemgetter
        newlist = sorted(result, key=self.sortkeypicker(tag))
        return newlist

    def decimalprec(self, num):
        return "{0:.2f}".format(num)

    def sellbuy(self, st1, st2):
        grp = st1["symbol"]
        quant = min(st1["quantity"], st2["quantity"])
        if st2["type"] == "M":
            price = st1["price"]
        else:
            price = st2["price"]
        return ('{sym}|{sell_id},{sell_type},{quant},{cost}|{cost},{quant},{buy_type},{buy_id}'.format(
            sym=st1["symbol"], sell_id=st1["id"], sell_type=st1[
                "type"], quant=quant, cost=self.decimalprec(price),
            buy_type=st2["type"], buy_id=st2["id"]
        ))

    def querysellbuy(self, st1, st2):
        if st1 == None:
            return ('{sym}||{cost2},{quant2},{buy_type},{buy_id}'.format(
                sym=st2["symbol"],
                quant2=st2["quantity"], cost2=self.decimalprec(st2["price"]), buy_type=st2["type"], buy_id=st2["id"]
            ))
        if st2 == None:
            return ('{sym}|{sell_id},{sell_type},{quant1},{cost1}|'.format(
                sym=st1["symbol"], sell_id=st1["id"], sell_type=st1["type"], quant1=st1[
                    "quantity"], cost1=self.decimalprec(st1["price"])
            ))
        return ('{sym}|{sell_id},{sell_type},{quant1},{cost1}|{cost2},{quant2},{buy_type},{buy_id}'.format(
            sym=st1["symbol"], sell_id=st1["id"], sell_type=st1["type"], quant1=st1[
                "quantity"], cost1=self.decimalprec(st1["price"]),
            quant2=st2["quantity"], cost2=self.decimalprec(st2["price"]), buy_type=st2["type"], buy_id=st2["id"]
        ))

    def cancel(self, arr_id, timestamp):
        res = []
        try:
            if self.weirdquery(int(timestamp)):
                return res
            stock = self.get_stock(arr_id)
            if stock["closed"] == False:
                stock["closed"] = True
                self.update(stock)
                res.append(
                    ('{order_id} - CancelAccept').format(order_id=arr_id))
            else:
                res.append(
                    ('{order_id} - CancelReject - 404 - Order does not exist').format(order_id=arr_id))
        except Exception as e:
            res.append(
                ('{order_id} - CancelReject - 404 - Order does not exist').format(order_id=arr_id))
        return res

    def add_none(self, lst, leng):
        while leng > 0:
            lst.append(None)
            leng -= 1
        return lst

    def state(self, time, sym):
        res = []
        keys = list(self.buy.keys()) + list(self.sell.keys())
        keys = list(set(keys))
        keys.sort()
        if sym != "":
            keys = [sym]
        for key in keys:
            buy = []
            sell = []
            if key in self.buy:
                buy = self.buy[key].copy()
            if key in self.sell:
                sell = self.sell[key].copy()
            buy = self.return_stocks_list(
                buy, ["-market", "-price", "time", "fifo"])
            sell = self.return_stocks_list(
                sell, ["-market", "price", "time", "fifo"])
            buy = buy[:5]
            sell = sell[:5]
            l = max(len(buy), len(sell))
            if len(buy) < len(sell):
                buy = self.add_none(buy, len(sell) - len(buy))
            elif len(buy) > len(sell):
                sell = self.add_none(sell, len(buy) - len(sell))
            for x, y in zip(buy, sell):
                if x == None and y == None:
                    continue
                res.append(self.querysellbuy(x, y))
        # print("here")
        # print(sym)
        # print(time)
        # print(res)
        return res

    def sortkeypicker(self, keynames):
        negate = set()
        for i, k in enumerate(keynames):
            if k[:1] == '-':
                keynames[i] = k[1:]
                negate.add(k[1:])

        def getit(adict):
            composite = [adict[k] for k in keynames]
            for i, (k, v) in enumerate(zip(keynames, composite)):
                if k in negate:
                    composite[i] = -v
            return composite
        return getit

    def match(self, time, symbol=None):
        res = []
        if self.weirdquery(int(time)):
            return res
        sym_arr = list(self.buy.keys())
        sym_arr.sort()
        val_symbol = symbol
        if symbol != None:
            sym_arr = [symbol]
        for symbol in sym_arr:
            stocks = self.buy[symbol]
            stocks = self.return_stocks_list(
                stocks, ["-market", "-price", "time", "fifo"])
            related_stocks = []
            if symbol in self.sell.keys():
                related_stocks = self.sell[symbol]
            related_stocks = self.return_stocks_list(
                related_stocks, ["-market", "price", "time", "fifo"])
            for stock in stocks:
                # print(stock)
                prc = float(sys.maxsize)
                if stock["market"] == 0:
                    prc = stock["price"]
                buy_quant = stock["quantity"]
                for sell in related_stocks:
                    # print("sell")
                    # print(sell)
                    if prc >= sell["price"] and stock["quantity"] > 0 and sell["quantity"] > 0:
                        res.append(self.sellbuy(stock, sell))
                        selling_quantity = min(buy_quant, sell["quantity"])
                        sell["quantity"] -= selling_quantity
                        stock["quantity"] -= selling_quantity
                        buy_quant = stock["quantity"]
                        if sell["quantity"] == 0:
                            sell["closed"] = True
                        self.update(sell)
                if stock["quantity"] == 0:
                    stock["closed"] = True
                    self.update(stock)
        self.closeIOC(val_symbol)
        return res

    def closeIOC(self, symbol=None):
        for stock in self.stock:
            if (stock["type"] == 'I' and (stock["symbol"] == symbol or symbol == None)):
                stock["closed"] = True
        return

    def storesQ(self, queries):
        count = 0
        for query in queries:
            count += 1
            arr = query.split(',')
            if arr[0] != 'Q':
                continue
            self.queriesQ[count] = query
            if len(arr) >= 2:
                y1 = arr[1]
                y2 = ""
                if len(arr) >= 3:
                    y2 = arr[2]
                    if len(y2) != 3:
                        y1 = y2
                if len(y1) != 3:
                    y1 = int(y1)
                    if y1 not in self.timesQ:
                        self.timesQ[y1] = [count]
                    else:
                        self.timesQ[y1].append(count)
        return

    def find_q_parameters(self, arr):
        y1 = ""
        y2 = ""
        if len(arr) >= 2:
            y1 = arr[1]
            y2 = ""
            if len(arr) >= 3:
                y2 = arr[2]
            #print(y1,y2)
            if len(y1) != 3:
                y1 = int(y1)
            else:
                temp = y2
                y2 = y1
                if temp != "":
                    y1 = int(temp)
                else:
                    y1 = self.time
        return self.state(y1, y2)

    def find_Qtime(self, arr):
        try:
            time = 0
            if arr[0] == 'N' or arr[0] == 'A' or arr[0] == 'X':
                time = int(arr[2])
            elif arr[0] == 'M':
                time = int(arr[1])
            res = []
            for k in self.timesQ.keys():
                k = int(k)
                #print(k, time)
                if k < time:
                    for query_id in self.timesQ[k]:
                        if query_id not in self.ansQ:
                            query = self.queriesQ[query_id]
                            new_arr = query.split(',')
                            self.ansQ[query_id] = self.find_q_parameters(
                                new_arr)
        except Exception as e:
            return
# Complete the function below.


def processQueries(queries):
    stock = stocks()
    result = []
    stock.storesQ(queries)
    count = 0
    #print(stock.timesQ)
    # print(stock.timesQ)
    # print(stock.queriesQ)
    # Write your code here.
    for query in queries:
        count += 1
        # print(count)
        # print(stock.ansQ)
        arr = query.split(',')
        stock.find_Qtime(arr)
        if count in stock.ansQ.keys():
            result += stock.ansQ[count]
            continue
        if arr[0] == 'N':
            if len(arr) == 8:
                result += stock.add(arr[1], arr[2], arr[3],
                                    arr[4], arr[5], arr[6], arr[7], count)
            else:
                result += [("{order} - Reject - 303 - Invalid order details".format(order=arr[1]))]
        elif arr[0] == 'A':
            if len(arr) == 8:
                result += stock.append(arr[1], arr[2],
                                       arr[3], arr[4], arr[5], arr[6], arr[7])
            else:
                result += [
                    "{order} - AmendReject - 101 - Invalid amendment details".format(order=arr[1])]
        elif arr[0] == 'M':
            sym = None
            if len(arr) >= 3:
                sym = arr[2]
            result += stock.match(arr[1], sym)
        elif arr[0] == 'X':
            result += stock.cancel(arr[1], arr[2])

        elif arr[0] == 'Q':
            # print(count)
            result += stock.find_q_parameters(arr)

    return result

if __name__ == '__main__':
    #f = open(os.environ['OUTPUT_PATH'], 'w')

    queries_size = int(input())

    queries = []
    for _ in range(queries_size):
        queries_item = input()
        queries.append(queries_item)

    response = processQueries(queries)
    print(*response, sep='\n')
    # f.write("\n".join(response))

    # f.write('\n')

    # f.close()
