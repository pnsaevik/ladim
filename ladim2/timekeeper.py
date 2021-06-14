"""Time related information for LADiM"""

# ================================
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# 2020-10-29
# ================================

import logging
import datetime
import re
from typing import Union, Optional, Dict, Any, List
import numpy as np

Time = Union[str, np.datetime64, datetime.datetime]
TimeDelta = Union[int, np.timedelta64, datetime.timedelta, List[Union[int, str]], str]


DEBUG = False

logger = logging.getLogger(__name__)
if DEBUG:
    logger.setLevel(logging.DEBUG)


class TimeKeeper:
    """Time utilities for LADiM

    attributes:
        start_time: start time of simulation
        stop_time: stop time for simulation
        time_reversal: flag for time reversal
        initial_time: usually start_time, stop_time if time_reversal
        reference_time: reference time for cf-standard output
        time_reversed: switch for time reversal
        dt: seconds, model time step
        Nsteps: Total number of time steps in simulation

    methods:
        time2step: time -> time step number
        step2isotime: time step number -> yyyy-mm-ddThh:mm:ss
        step2nctime: time -> seconds since reference time (or hours/days)
        cfunits: reference time -> "seconds since reference time" (or hours/days)

    """

    unit_table = dict(s="seconds", m="minutes", h="hours", d="days")

    def __init__(
        self,
        start: Time,
        stop: Time,
        dt: TimeDelta,
        reference: Optional[Time] = None,
        time_reversal: bool = False,
        modules: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        start      start time
        stop       stop time
        dt         duration of time step
        reference  reference time (for netcdf), optional, default=start

        start, stop, reference given as iso-strings or datetime instances
        dt given i seconds

        """

        logger.info("Initiating the timekeeper")
        self.modules = modules
        self.start_time = np.datetime64(start, "s")
        self.stop_time = np.datetime64(stop, "s")
        self.time_reversal = time_reversal
        self.time = self.start_time  # Running clock
        self.step = 0
        logger.info("  Model start time: %s", self.start_time)
        logger.info("  Model stop time: %s", self.stop_time)

        # Quality control
        duration = self.stop_time - self.start_time
        if time_reversal != (duration < np.timedelta64(0)):
            if time_reversal:
                print("ERROR: Backwards time and start before stop")
            else:
                print("ERROR: Forward time and stop before start")
            raise SystemExit(3)

        self.min_time = min(self.start_time, self.stop_time)
        self.max_time = max(self.start_time, self.stop_time)

        if reference:
            self.reference_time = np.datetime64(reference, "s")
        else:
            self.reference_time = self.min_time
        logger.info("  Reference time: %s", self.reference_time)

        self.dt = normalize_period(dt)  # np.timedelta64(-,"s")
        self.dtsec = self.dt / np.timedelta64(1, "s")  # seconds
        logger.info("  Time step: %s", self.dt)

        # Number of time steps (excluding initial)
        self.Nsteps = int(abs(duration) // self.dt)
        self.simulation_time = self.Nsteps * self.dt
        logger.info("  Length of  simulation: %s", duration2iso(duration))
        logger.info("  Number of time steps: %d", self.Nsteps)

    def update(self) -> None:
        """Update the clock"""
        self.step += 1
        if self.time_reversal:
            self.time = self.time - self.dt
        else:
            self.time = self.time + self.dt

    def reset(self) -> None:
        """Reset the clock"""
        self.time = self.start_time

    def nctime(self, unit: str = "s") -> float:
        """Get float value of model time"""
        delta = self.time - self.reference_time
        return float(delta / np.timedelta64(1, unit))

    def time2step(self, time_: Time) -> int:
        """Timestep from time

        time can be datetime instance or an iso time string
        """
        if self.time_reversal:
            return int((self.start_time - np.datetime64(time_)) // self.dt)
        return int((np.datetime64(time_) - self.start_time) // self.dt)

    def step2isotime(self, stepnr: int) -> str:
        """Return time in iso 8601 format from a time step number"""
        if self.time_reversal:
            return str(self.start_time - stepnr * self.dt)
        return str(self.start_time + stepnr * self.dt)

    def step2nctime(self, stepnr: int, unit: str = "s") -> float:
        """
        Return value from a time step following the netcdf standard

        unit should be a single character, "s", "m", or "h", default = "s"
        """
        if self.time_reversal:
            delta = self.start_time - stepnr * self.dt - self.reference_time
        else:
            delta = self.start_time + stepnr * self.dt - self.reference_time
        return float(delta / np.timedelta64(1, unit))

    def cf_units(self, unit: str = "s") -> str:
        """Return string with units for time following the CF standard"""
        return f"{self.unit_table[unit]} since {self.reference_time}"


def duration2iso(duration: Union[datetime.timedelta, np.timedelta64]) -> str:
    """Convert time delta to ISO 8601 format"""
    if isinstance(duration, np.timedelta64):
        seconds = int(duration / np.timedelta64(1, "s") + 0.5)
    else:
        seconds = int(duration.total_seconds() + 0.5)

    if not seconds:  # zero duration
        return "PT0S"

    days, sec = divmod(seconds, 86400)
    T = "T" if sec else ""
    hours, sec = divmod(sec, 3600)
    min_, sec = divmod(sec, 60)
    daystr = f"{days:d}D" if days else ""
    hourstr = f"{hours:d}H" if hours else ""
    minstr = f"{min_:d}M" if min_ else ""
    secstr = f"{sec:d}S" if sec else ""
    if duration:
        return "".join(("P", daystr, T, hourstr, minstr, secstr))
    return "PT0S"  # else zero duration


def normalize_period(per: TimeDelta) -> np.timedelta64:
    """Normalize different time period formats to np.timedelta64(-,"s")

    Accepted formats:
       int:  number of seconds
       np.timedelta64
       [value, unit]:  np.timedelta64(value, unit), unit = "h", "m", "s"
       ISO 8601 format: "PTxHyMzS", x hours, y minutes, z seconds
    """

    if isinstance(per, (int, np.timedelta64, datetime.timedelta)):
        return np.timedelta64(per, "s")
    # if isinstance(per, np.timedelta64):
    #    return per.astype("m8[s]")

    if isinstance(per, list):  # [value, unit] from yaml
        try:
            value, unit = per
            if isinstance(value, int) and isinstance(unit, str):
                return np.timedelta64(np.timedelta64(value, unit), "s")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{per} is not a valid time period") from exc

    if isinstance(per, str):  # ISO 8601 standard PTxHyMzS
        pattern = r"^PT(\d+H)?(\d+M)?(\d+S)?$"
        m = re.match(pattern, per)
        if m is None:
            raise ValueError(f"{per} is not a valid time period")
        td = np.timedelta64(0, "s")
        if not any(m.groups()):
            raise ValueError(f"{per} is not a valid time period")
        for item in m.groups():
            if item:
                value = int(item[:-1])
                unit = item[-1].lower()
                td = td + np.timedelta64(value, unit)
        return td

    # None of the above
    raise ValueError(f"{per} is not a valid time period")
