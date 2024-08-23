
import pynlin.fiber
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
import pynlin.constellations

def do_time_integrals(fiber, wdm, pulse, pulse_shape="Nyquist"):
  partial_collisions_margin = 5
  points_per_collision = 10

  a_channels = [(0,0)]
  pynlin.nlin.iterate_time_integrals(
      wdm,
      fiber,
      a_channels[0],
      pulse,
      "results/general_results_alt.h5",
      points_per_collision=points_per_collision, # kwargs
      use_multiprocessing=True,
      partial_collisions_margin=partial_collisions_margin,
      speedup_pulse_propagation=True
  )
  return 0