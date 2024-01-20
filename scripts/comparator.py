from space_integrals_general import *
from time_integrals import do_time_integrals

f = open("./scripts/sim_config.json")
data = json.load(f)
num_channels = data["num_channels"] # 3 channels
interfering_channels = [list(range(num_channels-1))[0]]
length = 1e3 * 200
do_time_integrals(length)
compare_interferent(interfering_channels=interfering_channels)