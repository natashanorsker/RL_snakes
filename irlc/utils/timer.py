"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from collections import defaultdict
import datetime

class Timer:
    def __init__(self, start=False):
        self.tspend = defaultdict(lambda: 0)
        self.t_start = {}
        self.s_ = None
        if start:
            self.start()

    def start(self):
        self.s_ = datetime.datetime.now()

    def tic(self, name):
        self.lst = name
        self.t_start[name] = datetime.datetime.now()

    def toc(self, name=None):
        name = name if name is not None else self.lst
        self.tspend[name] += (datetime.datetime.now() - self.t_start[name]).total_seconds()

    def display(self):
        Tknown = sum(self.tspend.values())
        if self.s_ is not None:
            Ttot = (datetime.datetime.now() - self.s_).total_seconds()
        s = ", ".join( [f"{k}: {v:.2f} ({int(v/Tknown*100)} %)" for k, v in self.tspend.items()] )
        if self.s_ is not None:
            return f"{Ttot:.2f} ({(Tknown/Ttot*100):.1f} %). " + s
        else:
            return s
