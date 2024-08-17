
import pynlin.fiber
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
import pynlin.constellations

def do_time_integrals(fiber, wdm, baud_rate, pulse_shape="Nyquist"):
  partial_collision_margin = 5
  points_per_collision = 10

  coi_index = [0]
  pynlin.nlin.iterate_time_integrals(
      baud_rate,
      wdm,
      coi_index,
      fiber,
      fiber.length,
      "results/general_results_alt.h5",
      pulse_shape=pulse_shape,
      rolloff_factor=0.0,
      samples_per_symbol=10,
      points_per_collision=points_per_collision,
      use_multiprocessing=True,
      partial_collisions_start=partial_collision_margin,
      partial_collisions_end=partial_collision_margin,
  )
  return 0