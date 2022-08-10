import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import h5py
import math
import os
from scipy.interpolate import interp1d
import tqdm
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
from pynlin.fiber import Fiber
from pynlin.utils import dBm2watt, watt2dBm
from pynlin.wdm import WDM
import pynlin.constellations
from matplotlib.patches import Arc
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument(
    "-L",
    "--fiber-length",
    default=80,
    type=float,
    help="The length of the fiber in kilometers.",
)
args = parser.parse_args()


plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '500'
plt.rcParams['font.size'] = '14'


###############################
#### fiber length setup #######
###############################
length_setup = int(args.fiber_length)
fiber_length = length_setup * 1e3


plot_save_path = '/home/lorenzi/Scrivania/tesi/tex/images/classical/'+str(length_setup)+'km/'
results_path = '../results_'+str(length_setup)+'/'



# PLOTTING PARAMETERS
interfering_grid_index = 1
#power_dBm_list = [-20, -10, -5, 0]
#power_dBm_list = np.linspace(-20, 0, 3)
power_dBm_list = [0.0, -10.0, -20.0]

arity_list = [16]

wavelength = 1550
baud_rate = 10
dispersion = 18
channel_spacing = 100
num_channels = 50
baud_rate = baud_rate * 1e9

beta2 = pynlin.utils.dispersion_to_beta2(
    dispersion * 1e-12 / (1e-9 * 1e3), wavelength * 1e-9
)
fiber = pynlin.fiber.Fiber(
    effective_area=80e-12,
    beta2=beta2
)
wdm = pynlin.wdm.WDM(
    spacing=channel_spacing,
    num_channels=num_channels,
    center_frequency=190
)
partial_collision_margin = 5
points_per_collision = 10 

print("beta2: ", fiber.beta2)
print("gamma: ", fiber.gamma)

Delta_theta_2_co = np.zeros_like(
    np.ndarray(shape=(len(power_dBm_list), len(arity_list)))
    )
Delta_theta_2_cnt = np.zeros_like(Delta_theta_2_co)

show_plots = False
show_pumps = False

if input("\nASE-Signal_pump profile plotter: \n\t>Length= "+str(length_setup)+"km \n\t>power list= "+str(power_dBm_list)+"\nAre you sure? (y/[n])") != "y":
    exit()

for idx, power_dBm in enumerate(power_dBm_list):
    average_power =  dBm2watt(power_dBm)

    # SIMULATION DATA LOAD =================================
    pump_solution_co = np.load(results_path + 'pump_solution_co_' + str(power_dBm) + '.npy')
    signal_solution_co = np.load(results_path + 'signal_solution_co_' + str(power_dBm) + '.npy')
    ase_solution_co = np.load(results_path + 'ase_solution_co_' + str(power_dBm) + '.npy')

    pump_solution_cnt =    np.load(results_path + 'pump_solution_cnt_' + str(power_dBm) + '.npy')
    signal_solution_cnt =  np.load(results_path + 'signal_solution_cnt_' + str(power_dBm) + '.npy')
    ase_solution_cnt = np.load(results_path + 'ase_solution_cnt_' + str(power_dBm) + '.npy')

    pump_solution_bi =    np.load(results_path + 'pump_solution_bi_' + str(power_dBm) + '.npy')
    signal_solution_bi =  np.load(results_path + 'signal_solution_bi_' + str(power_dBm) + '.npy')
    ase_solution_bi = np.load(results_path + 'ase_solution_bi_' + str(power_dBm) + '.npy')

    #z_max = np.load(results_path + 'z_max.npy')
    #f = h5py.File(results_path + '0_9_results.h5', 'r')
    z_max = np.linspace(0, fiber_length, np.shape(pump_solution_cnt)[0])
    z_max *= 1e-3
    custom_lines = [Line2D([0], [0], color="k", lw=2),
                    Line2D([0], [0], color="b", lw=2)]
    if show_pumps:
        custom_lines.append(Line2D([0], [0], color="r", lw=2))
    plt.figure(figsize=(8, 6))
    plt.plot(z_max, np.transpose(watt2dBm([ase_solution_co[:, idx] for idx in [0, 24, 49]])), color="k")
    plt.plot(z_max,  np.transpose(watt2dBm([signal_solution_co[:, idx] for idx in [0, 24, 49]])), color="b")
    if show_pumps:
        plt.plot(z_max,watt2dBm(pump_solution_co), color="r")

    plt.legend(custom_lines, ['ASE', 'Signal', 'Pump'])
    x_annotation = 100 
    for chan in [0, 49]:
        plt.annotate("ch."+str(chan+1), (x_annotation/len(z_max)*z_max[len(z_max)-1], watt2dBm(ase_solution_co[x_annotation, chan]) -1 - 3*np.sign(25-chan)))
        plt.annotate("ch."+str(chan+1), (x_annotation/len(z_max)*z_max[len(z_max)-1], watt2dBm(signal_solution_co[x_annotation, chan]) -1 - 3*np.sign(25-chan)))


    plt.xlabel("z [km]")
    plt.ylabel("Wave power [dBm]")
    plt.grid("on")
    plt.minorticks_on()
    osnr = 10*np.log10(signal_solution_co[-1:, :]/ase_solution_co[-1:, :])
    print(osnr)
    # plt.figure()
    # plt.plot(signal_wavelengths, watt2dBm(signal_solution[-1]), color="k")
    if show_plots:
        plt.show()

    plt.tight_layout()
    plt.savefig(plot_save_path+"profile"+str(power_dBm)+ "_co.pdf")

    plt.figure()
    plt.plot(z_max, np.transpose(watt2dBm([ase_solution_cnt[:, idx] for idx in [0, 24, 49]])), color="k")
    plt.plot(z_max,  np.transpose(watt2dBm([signal_solution_cnt[:, idx] for idx in [0, 24, 49]])), color="b")
    if show_pumps:
        plt.plot(z_max,watt2dBm(pump_solution_cnt), color="r")

    plt.legend(custom_lines, ['ASE', 'Signal', 'Pump'])
    for chan in [0, 49]:
        plt.annotate("ch."+str(chan+1), (x_annotation/len(z_max)*z_max[len(z_max)-1], watt2dBm(ase_solution_cnt[x_annotation, chan]) -1 - 3*np.sign(25-chan)))
        plt.annotate("ch."+str(chan+1), (x_annotation/len(z_max)*z_max[len(z_max)-1], watt2dBm(signal_solution_cnt[x_annotation, chan]) -1 - 3*np.sign(25-chan)))

    plt.xlabel("z [km]")
    plt.ylabel("Wave power [dBm]")
    plt.grid("on")
    plt.minorticks_on()
    osnr = 10*np.log10(signal_solution_cnt[-1:, :]/ase_solution_cnt[-1:, :])
    print(osnr)
    # plt.figure()
    # plt.plot(signal_wavelengths, watt2dBm(signal_solution[-1]), color="k")
    if show_plots:
        plt.show()

    plt.tight_layout()
    plt.savefig(plot_save_path+"profile"+str(power_dBm)+ "_cnt.pdf")

    plt.figure()
    plt.plot(z_max, np.transpose(watt2dBm([ase_solution_bi[:, idx] for idx in [0, 24, 49]])), color="k")
    plt.plot(z_max,  np.transpose(watt2dBm([signal_solution_bi[:, idx] for idx in [0, 24, 49]])), color="b")
    if show_pumps:
        plt.plot(z_max, watt2dBm(pump_solution_bi), color="r")
    

    plt.legend(custom_lines, ['ASE', 'Signal', 'Pump'])
    for chan in [0, 49]:
        plt.annotate("ch."+str(chan+1), (x_annotation/len(z_max)*z_max[len(z_max)-1], watt2dBm(ase_solution_bi[x_annotation, chan]) -1 - 3*np.sign(25-chan)))
        plt.annotate("ch."+str(chan+1), (x_annotation/len(z_max)*z_max[len(z_max)-1], watt2dBm(signal_solution_bi[x_annotation, chan]) -1 - 3*np.sign(25-chan)))

    plt.xlabel("z [km]")
    plt.ylabel("Wave power [dBm]")
    plt.grid("on")
    plt.minorticks_on()
    osnr = 10*np.log10(signal_solution_bi[-1:, :]/ase_solution_bi[-1:, :])
    print(osnr)
    # plt.figure()
    # plt.plot(signal_wavelengths, watt2dBm(signal_solution[-1]), color="k")
    if show_plots:
        plt.show()

    plt.tight_layout()
    plt.savefig(plot_save_path+"profile"+str(power_dBm)+ "_bi.pdf")