"""
Timer utility classes for benchmarking and wall-time measurement.
"""

import time

import torch


class Timer:
    """
    Base timer interface for benchmarking (abstract base).
    """
    @property
    def elapsed_time(self):
        """
        Return the measured elapsed time.
        """
        raise NotImplementedError

    def pause(self):
        """
        Pause the timer (if supported).
        """
        raise NotImplementedError

    def resume(self):
        """
        Resume the timer after a pause.
        """
        raise NotImplementedError

    def __float__(self):
        """
        Return timer value as float (seconds).
        """
        return float(self.elapsed_time)

    def __coerce__(self, other):
        """
        Coerce timer to float for numeric operations.
        """
        return float(self), other

    def __str__(self):
        """
        String representation of timer value.
        """
        return str(float(self))

    def __format__(self, format_spec):
        """
        Custom string formatting for timer.
        """
        return f"{float(self) :{format_spec}}"

    def __repr__(self):
        """
        Repr representation of timer value.
        """
        return str(float(self))


class CPUTimer(Timer):
    """
    Context manager to measure CPU time.

    Usage::

        >>> with CPUTimer() as t:
        >>>      time.sleep(2)
        >>> t.elapsed_time # 2 seconds, approximately
    """

    def __init__(self, buffer_size: int = None):
        self.start_time_ll = []
        self.end_time_ll = []
        self.is_paused = False

    def __enter__(self):
        self.start_time_ll.append(time.perf_counter())

        return self

    def pause(self):
        assert not self.is_paused
        self.end_time_ll.append(time.perf_counter())
        self.is_paused = True

    def resume(self):
        assert self.is_paused
        self.start_time_ll.append(time.perf_counter())
        self.is_paused = False

    def __exit__(self, type, value, traceback):
        if self.is_paused:
            self.resume()
        self.end_time_ll.append(time.perf_counter())

    @property
    def elapsed_time(self):
        if len(self.end_time_ll) == 0:
            raise RuntimeError("No end times recorded.")

        assert len(self.start_time_ll) == len(self.end_time_ll)
        time_ll = [e - s for s, e in zip(self.start_time_ll, self.end_time_ll)]
        return sum(time_ll)


class CudaTimer(Timer):
    """
    Context manager to measure Cuda time.
    Example ::

        >>> with CudaTimer() as t:
        >>>      time.sleep(2)
        >>> t.elapsed_time # 2 ish, highest precision afforded by your machine

    Reference: https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
    """

    def __init__(self, buffer_size: int = 1):
        # Buffering cuda events can help reduce their overhead
        torch.cuda.synchronize()

        self.start_event_ll = [
            torch.cuda.Event(enable_timing=True) for _ in range(buffer_size)
        ]
        self.end_event_ll = [
            torch.cuda.Event(enable_timing=True) for _ in range(buffer_size)
        ]
        self.event_idx = 0
        self.is_paused = False

    def __enter__(self):
        self.start_event_ll[0].record()
        return self

    def pause(self):
        assert not self.is_paused
        self.end_event_ll[self.event_idx].record()
        self.event_idx += 1

        if self.event_idx == len(self.end_event_ll):
            # Buffer size reached
            self.end_event_ll.append(torch.cuda.Event(enable_timing=True))
            self.start_event_ll.append(torch.cuda.Event(enable_timing=True))

        self.is_paused = True

    def resume(self):
        assert self.is_paused
        self.start_event_ll[self.event_idx].record()
        self.is_paused = False

    def __exit__(self, type, value, traceback):
        if self.is_paused:
            self.resume()

        self.end_event_ll[self.event_idx].record()

    @property
    def elapsed_time(self):
        """
        Call this method only at the end. For excluding code snippets, use pause and resume.
        :return:
        """
        torch.cuda.synchronize()
        # Reported in milliseconds
        time_ll = [
            self.start_event_ll[idx].elapsed_time(self.end_event_ll[idx]) * 1e-3
            for idx in range(self.event_idx + 1)
        ]
        return sum(time_ll)


if __name__ == "__main__":
    with CPUTimer() as t:
        time.sleep(1)

        t.pause()
        time.sleep(3)
        t.resume()

        time.sleep(1.5)

    print(f"Elapsed CPU time {t}")

    if torch.cuda.is_available():
        with CudaTimer(buffer_size=1) as t:
            time.sleep(1)

            t.pause()
            # Will not include this in timing
            time.sleep(3)

            t.resume()
            time.sleep(1)

    print(f"Elapsed CUDA time {t}")
