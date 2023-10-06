#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import threading


class ReadLock:
    def __init__(self, size):
        self.size = size
        self.count = 0
        self.mutex = threading.Lock()
        self.not_full = threading.Condition(self.mutex)
        self.not_empty = threading.Condition(self.mutex)

    def acquire(self):
        with self.not_full:
            while self.count >= self.size:
                self.not_full.wait()
            self.count += 1
            self.not_empty.notify()

    def release(self):
        with self.not_empty:
            while self.count <= 0:
                self.not_empty.wait()
            self.count -= 1
            self.not_full.notify()
