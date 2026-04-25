"""
Profiling decorators and resettable profiler utilities for line-by-line and function timing.
"""
import atexit
from functools import wraps
from typing import Optional

from line_profiler import LineProfiler


# Makes the line profiler decorator
# available
# See: https://lothiraldan.github.io/2018-02-18-python-line-profiler-without-magic/


class ResettableProfiler:
    """
    Profiling helper to wrap, track, and benchmark arbitrary Python callable objects.
    Use `@profile` to annotate functions, and manage benchmarks with start()/stop()/reset().
    """
    def __init__(self):
        """
        Initialize profiling state and tracked functions.
        """
        self.functions: list[list] = []
        self.line_profiler: Optional[LineProfiler] = None

    def __call__(self, func):
        """
        Register a function for profiling decorator support.

        :param func: Target python function
        :return: Profile-wrapped function
        """
        index = len(self.functions)

        @wraps(func)
        def wrap(*args, **kw):
            return self.functions[index][1](*args, **kw)

        self.functions.append([func, func])
        return wrap

    def start(self):
        """
        Start a line profiler for all tracked functions.
        """
        self.line_profiler = LineProfiler()
        for f in self.functions:
            f[1] = self.line_profiler(f[0])

    def stop(self, *, print: bool = True):
        """
        Stop line profiling and optionally print stats.

        :param print: If True, print stats
        """
        for f in self.functions:
            f[1] = f[0]
        if self.line_profiler and print:
            self.line_profiler.print_stats()

    def reset(self):
        """
        Restart profiler (stops then starts again).
        """
        self.stop(print=False)
        self.start()


profiler = LineProfiler()
resettable_profiler = ResettableProfiler()

# Before exiting print stats
atexit.register(profiler.print_stats)
