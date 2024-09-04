import numpy as np
from typing import Generator, List, Tuple
from pynlin.fiber import Fiber
import math
from pynlin.fiber import Fiber, SMFiber, MMFiber
from pynlin.wdm import WDM
from pynlin.pulses import Pulse

def get_interfering_channels(): 
  pass

def get_interfering_frequencies(
    channel_of_interest: float,
    frequency_grid: np.ndarray,
) -> List[float]:
    """Given a channel frequency and an  iterable of frequencies, generate a
    list of interfering channel frequencies."""
    combinations = []
    for x in frequency_grid:
        if x != channel_of_interest:
            combinations.append(x)
    return combinations

def get_frequency_spacing(a_chan, b_chan, wdm):
  frequency_grid = wdm.frequency_grid()
  return frequency_grid[b_chan[1]] - frequency_grid[a_chan[1]]

def get_m_values(
    fiber: Fiber,
    wdm: WDM,
    a_chan: Tuple[int, int],
    b_chan: Tuple[int, int],
    T: float,
    partial_collisions_start:int,
) -> np.ndarray:
    """Get values of the m indeces to compute the X0mm XPM coefficients for.

    Computes those indeces for which the collisions fall inside the
    fiber. By default, 10 extra partial collisions at each end of the
    fiber are computed. This parameter can be controlled by the
    `partial_collisions_start` and `partial_collisions_end` kwargs.
    """
    partial_collisions_end = partial_collisions_start
    dgd = get_dgd(a_chan, b_chan, fiber, wdm)
    m_max = -(fiber.length * dgd) / T
    # print(m_max)
    # print(-m_max * T /dgd)
    if m_max < 0:
        m_max = math.ceil(m_max)
        return np.arange(m_max - partial_collisions_start, partial_collisions_end + 1)
    else:
        m_max = math.floor(m_max)
        return np.arange(-partial_collisions_start, m_max + partial_collisions_end + 1)


def get_collision_location(m, 
                           fiber: Fiber, 
                           wdm: WDM, 
                           a_chan:Tuple[int, int], 
                           b_chan: Tuple[int, int],
                           pulse: Pulse) -> float:
    """For the specified index m, compute the position of the corresponding
    complete collision."""
    dgd = get_dgd(a_chan, b_chan, fiber, wdm)
    # print(f" DGD: {dgd}, a_chan: {a_chan}, b_chan: {b_chan}")
    return -m / (pulse.baud_rate *  dgd)
  
  
def get_dgd(a_chan, b_chan, fiber, wdm) -> float:
    freq_grid = wdm.frequency_grid()
    if isinstance(fiber, SMFiber):
        assert (a_chan[0] == 0 and b_chan[0] == 0)
        return fiber.beta2 * 2 * np.pi * (freq_grid[b_chan[1]] - freq_grid[a_chan[1]])
    elif isinstance(fiber, MMFiber):
        return fiber.group_delay.evaluate_beta1(b_chan[0], freq_grid[b_chan[1]]) \
      - fiber.group_delay.evaluate_beta1(a_chan[0], freq_grid[a_chan[1]])


def get_gvd(b_chan, fiber, wdm) -> float:
    if isinstance(fiber, SMFiber):
        return fiber.beta2
    elif isinstance(fiber, MMFiber):
        return fiber.group_delay.evaluate_beta2(b_chan[0], wdm.frequency_grid()[b_chan[1]])
    pass
  
  
def get_z_walkoff(
  fiber: Fiber, 
  wdm: WDM, 
  a_chan: Tuple[int, int], 
  b_chan: Tuple[int, int], 
  pulse: Pulse):
  dgd = get_dgd(a_chan, b_chan, fiber, wdm)
  return np.abs(1/ (pulse.baud_rate * dgd))