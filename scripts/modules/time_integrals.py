
import pynlin.fiber
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
import pynlin.constellations
import json

def do_time_integrals(fiber, wdm, pulse_shape="Nyquist"):
  f = open("./scripts/sim_config.json")
  data = json.load(f)
  dispersion = data["dispersion"]
  baud_rate = data["baud_rate"]
  num_channels = data["num_channels"]
  channel_spacing = data["channel_spacing"]
  partial_collision_margin = data["partial_collision_margin"]
  wavelength = data["wavelength"]
  center_frequency = data["center_frequency"]

  channel_spacing = channel_spacing
  num_channels = num_channels
  baud_rate = baud_rate

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