"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import itertools
from heapq import heappush, heappop
REMOVED = '<removed-task>'  # placeholder for a removed task

class PriorityQueue:
    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.counter = itertools.count()  # unique sequence count

    def push(self, item, priority):
        'Add a new task or update the priority of an existing task'
        if item in self.entry_finder:
            remove_item(self.entry_finder, item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        heappush(self.pq, entry)

    def pop(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

    def isEmpty(self):
        return len(self.entry_finder) == 0

def remove_item(entry_finder, task):  # Can't recall why this is in seperate function. Makes little sense tbh.
    entry = entry_finder.pop(task)
    entry[-1] = REMOVED
