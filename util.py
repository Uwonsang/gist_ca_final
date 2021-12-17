import time

class time_measure():
    def __init__(self):
        self.s = 0
        self.e = 0
    
    def start(self):
        self.s = time.time()
        return 0
    
    def end(self):
        self.e = time.time() - self.s
        return self.e