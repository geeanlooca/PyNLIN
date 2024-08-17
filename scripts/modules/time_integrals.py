
import pynlin.fiber
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
import pynlin.constellations

def do_time_integrals(fiber, wdm, pulse, pulse_shape="Nyquist"):
  partial_collision_margin = 5
  points_per_collision = 10

  coi_index = [0]
  pynlin.nlin.iterate_time_integrals(
      wdm,
      coi_index,
      fiber,
      pulse,
      "results/general_results_alt.h5",
      points_per_collision=points_per_collision,
      use_multiprocessing=True,
      partial_collisions_start=partial_collision_margin,
      partial_collisions_end=partial_collision_margin,
  )
  return 0