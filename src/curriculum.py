#%%
import random

class Curriculum:
    def __init__(self, start, end, inc, interval):
        self.start = start
        self.end = end
        self.inc = inc
        self.interval = interval
        assert all(isinstance(x, int) for x in [start, end, inc, interval]), "All arguments must be integers"
        assert start <= end, "Start value must be less than end value"

        self.step_count = 0
        self._val = start

    @property
    def value(self):
        return self._val

    def update(self, step):
        if self.step_count < step:
            self.step_count = step
            
        if self.step_count % self.interval == 0 and self.step_count > 1:
            self._val += self.inc

        self._val =  min(self._val, self.end)
        return self._val


#%%

# c = Curriculum(0, 5, 4, 4)

# for step in range(0, 100, 1):
#     print(step, c.update(step))

# %%
