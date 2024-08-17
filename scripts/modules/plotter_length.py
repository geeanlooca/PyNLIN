import matplotlib.pyplot as plt
import os
import json

f = open("./scripts/sim_config.json")
data = json.load(f)
print(data)
dispersion=data["dispersion"] 
effective_area=data["effective_area"] 
baud_rate=data["baud_rate"] 
fiber_length=data["fiber_length"] 
num_channels=data["num_channels"] 
channel_spacing=data["channel_spacing"] 
center_frequency=data["center_frequency"] 
store_true=data["store_true"] 
pulse_shape=data["pulse_shape"] 
partial_collision_margin=data["partial_collision_margin"] 
num_co= data["num_co"] 
num_cnt=data["num_cnt"]
wavelength=data["wavelength"]

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '500'
plt.rcParams['font.size'] = '24'

length_setup = int(fiber_length*1e-3) 
plot_save_path = "/home/lorenzi/Scrivania/progetti/NLIN/plots_"+str(length_setup)+'/'+str(num_co)+'_co_'+str(num_cnt)+'_cnt/'
#
if not os.path.exists(plot_save_path):
    os.makedirs(plot_save_path)
#
results_path = '../results_'+str(length_setup)+'/'
results_path_bi = '../results_'+str(length_setup)+'/'+str(num_co)+'_co_'+str(num_cnt)+'_cnt/'
#
time_integrals_results_path = '../results/'