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
import matplotlib as mpl
import json

f = open("/home/lorenzi/Scrivania/progetti/NLIN/PyNLIN/scripts/sim_config.json")
data = json.load(f)
print(data)
dispersion=data["dispersion"] 
effective_area=data["effective_area"] 
baud_rate=data["baud_rate"] 
fiber_length=data["fiber_length"][0]
num_channels=data["num_channels"] 
channel_spacing=data["channel_spacing"] 
center_frequency=data["center_frequency"] 
store_true=data["store_true"] 
pulse_shape=data["pulse_shape"] 
partial_collision_margin=data["partial_collision_margin"] 
num_co= data["num_co"] 
num_cnt=data["num_cnt"]
wavelength=data["wavelength"]
special=data["special"]
pump_direction=data["pump_direction"]

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '500'
plt.rcParams['font.size'] = '24'

length_setup = int(fiber_length*1e-3) 
plot_save_path = "/home/lorenzi/Scrivania/progetti/NLIN/plots_"+str(length_setup)+'/'+str(num_co)+'_co_'+str(num_cnt)+'_cnt_'+special+'/'
#
if not os.path.exists(plot_save_path):
    os.makedirs(plot_save_path)
#
results_path = '../results_'+str(length_setup)+'/'
results_path_bi = '../results_'+str(length_setup)+'/'+str(num_co)+'_co_'+str(num_cnt)+'_cnt_'+special+'/'
#
time_integrals_results_path = '../results/'

# PLOTTING PARAMETERS
interfering_grid_index = 1
#power_dBm_list = [-20, -10, -5, 0]
power_dBm_list = [0.0, -10.0, -20.0]
power_dBm_list = np.linspace(-20, 0, 11)

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
signal_wavelengths = wdm.wavelength_grid()

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
linestyles = ["solid", "dashed", "dotted"]
coi_list = [0, 24, 49]
#if input("\nASE-Signal_pump profile plotter: \n\t>Length= " + str(length_setup) + "km \n\t>power list= " + str(power_dBm_list) + "\nAre you sure? (y/[n])") != "y":
 #   exit()
configs = [[num_co, num_cnt]]
for config in configs:
    num_co = config[0]
    num_cnt = config[1]
    for idx, power_dBm in enumerate(power_dBm_list):
        average_power = dBm2watt(power_dBm)
        results_path = '../results_' + str(length_setup) + '/'
        optimized_result_path = '../results_' + str(length_setup) + '/optimization/'
        optimized_result_path_bi =  '../results_' + str(length_setup) + '/optimization/'+ str(num_co) + '_co_' + str(num_cnt) + '_cnt/'
        results_path_bi = '../results_' + \
            str(length_setup) + '/' + str(num_co) + '_co_' + str(num_cnt) + '_cnt_'+special+'/'
        plot_save_path = "/home/lorenzi/Scrivania/progetti/NLIN/plots_"+str(length_setup)+'/'+str(num_co)+'_co_'+str(num_cnt)+'_cnt_'+special+'/'

        # SIMULATION DATA LOAD =================================

        pump_solution_co = np.load(
            results_path + 'pump_solution_co_' + str(power_dBm) + '.npy')
        signal_solution_co = np.load(
            results_path + 'signal_solution_co_' + str(power_dBm) + '.npy')
        ase_solution_co = np.load(
            results_path + 'ase_solution_co_' + str(power_dBm) + '.npy')

        pump_solution_cnt = np.load(
            results_path + 'pump_solution_cnt_' + str(power_dBm) + '.npy')
        signal_solution_cnt = np.load(
            results_path + 'signal_solution_cnt_' + str(power_dBm) + '.npy')
        ase_solution_cnt = np.load(
            results_path + 'ase_solution_cnt_' + str(power_dBm) + '.npy')

        pump_solution_bi = np.load(
            results_path_bi + 'pump_solution_bi_' + str(power_dBm) + '.npy')
        signal_solution_bi = np.load(
            results_path_bi + 'signal_solution_bi_' + str(power_dBm) + '.npy')
        ase_solution_bi = np.load(
            results_path_bi + 'ase_solution_bi_' + str(power_dBm) + '.npy')

        # plt.figure()
        # plt.plot(signal_wavelengths, watt2dBm(signal_solution_co[-1]), color="k")
        # plt.figure()
        # plt.plot(signal_wavelengths, watt2dBm(signal_solution_cnt[-1]), color="k")
        # plt.figure()
        # plt.plot(signal_wavelengths, watt2dBm(signal_solution_bi[-1]), color="k")

        #z_max = np.load(results_path + 'z_max.npy')
        #f = h5py.File(results_path + '0_9_results.h5', 'r')
        z_max = np.linspace(0, fiber_length, np.shape(pump_solution_bi)[0])
        z_max *= 1e-3
        custom_lines = [Line2D([0], [0], color="k", lw=2),
                        Line2D([0], [0], color="b", lw=2)]
        if show_pumps:
            custom_lines.append(Line2D([0], [0], color="r", lw=2))
       
        ########################################################################
        # SIGNALS AND ASE ONLY
        ########################################################################
        ##################
        fig1, (ax) = plt.subplots(nrows=1, figsize=(8, 6))

        plt.plot(z_max, np.transpose(
            watt2dBm([ase_solution_co[:, idx] for idx in [0, 24, 49]])), color="k")
        plt.plot(z_max, np.transpose(
            watt2dBm([signal_solution_co[:, idx] for idx in [0, 24, 49]])), color="b")
        if show_pumps:
            plt.plot(z_max, watt2dBm(pump_solution_co), color="r")

        plt.legend(custom_lines, ['ASE', 'Signal', 'Pump'])

        plt.xlabel("z [km]")
        plt.ylabel("Wave power [dBm]")
        plt.grid("on")
        plt.minorticks_on()
        osnr = 10 * np.log10(signal_solution_co[-1:, :] / ase_solution_co[-1:, :])
        # plt.figure()
        # plt.plot(signal_wavelengths, watt2dBm(signal_solution[-1]), color="k")
        if show_plots:
            plt.show()

        plt.tight_layout()
        plt.savefig(plot_save_path + "profile" + str(power_dBm) + "_co.pdf")

        ###############
        fig1, (ax) = plt.subplots(nrows=1, figsize=(8, 6))
        plt.plot(z_max, np.transpose(
            watt2dBm([ase_solution_cnt[:, idx] for idx in [0, 24, 49]])), color="k")
        plt.plot(z_max, np.transpose(
            watt2dBm([signal_solution_cnt[:, idx] for idx in [0, 24, 49]])), color="b")
        if show_pumps:
            plt.plot(z_max, watt2dBm(pump_solution_cnt), color="r")

        plt.legend(custom_lines, ['ASE', 'Signal', 'Pump'])


        plt.xlabel("z [km]")
        plt.ylabel("Wave power [dBm]")
        plt.grid("on")
        plt.minorticks_on()
        osnr = 10 * np.log10(signal_solution_cnt[-1:, :] / ase_solution_cnt[-1:, :])
        # plt.figure()
        # plt.plot(signal_wavelengths, watt2dBm(signal_solution[-1]), color="k")
        if show_plots:
            plt.show()

        plt.tight_layout()
        plt.savefig(plot_save_path + "profile" + str(power_dBm) + "_cnt.pdf")

        #########
        fig1, (ax) = plt.subplots(nrows=1, figsize=(8, 6))
        plt.plot(z_max, np.transpose(
            watt2dBm([ase_solution_bi[:, idx] for idx in [0, 24, 49]])), color="k")
        plt.plot(z_max, np.transpose(
            watt2dBm([signal_solution_bi[:, idx] for idx in [0, 24, 49]])), color="b")
        if show_pumps:
            plt.plot(z_max, watt2dBm(pump_solution_bi), color="r")

        plt.legend(custom_lines, ['ASE', 'Signal', 'Pump'])

        plt.xlabel("z [km]")
        plt.ylabel("Wave power [dBm]")
        plt.grid("on")
        plt.minorticks_on()
        osnr = 10 * np.log10(signal_solution_bi[-1:, :] / ase_solution_bi[-1:, :])
        # plt.figure()
        # plt.plot(signal_wavelengths, watt2dBm(signal_solution[-1]), color="k")
        if show_plots:
            plt.show()

        plt.tight_layout()
        plt.savefig(plot_save_path + "profile" + str(power_dBm) + "_bi.pdf")
    
        ########################################################################
        # SIGNALS ONLY
        ########################################################################
        labels = ["ch.1", "ch.25", "ch.50"]
        fig1, (ax) = plt.subplots(nrows=1, figsize=(8, 6))
        for ii, idx in enumerate(coi_list):
            plt.plot(z_max, np.transpose(watt2dBm(
                signal_solution_co[:, idx])), color="b", linestyle=linestyles[ii], label=labels[ii])
    

        plt.xlabel("z [km]")
        plt.ylabel("Signal power [dBm]")
        plt.grid("on")
        plt.legend()
        plt.minorticks_on()

        if show_plots:
            plt.show()

        plt.tight_layout()
        plt.savefig(plot_save_path + "signals" + str(power_dBm) + "_co.pdf")

        # à
        fig1, (ax) = plt.subplots(nrows=1, figsize=(8, 6))
        for ii, idx in enumerate(coi_list):
            plt.plot(z_max, np.transpose(watt2dBm(
                signal_solution_cnt[:, idx])), color="b", linestyle=linestyles[ii], label=labels[ii])

        plt.xlabel("z [km]")
        plt.ylabel("Signal power [dBm]")
        plt.grid("on")
        plt.legend()

        plt.minorticks_on()

        if show_plots:
            plt.show()

        plt.tight_layout()
        plt.savefig(plot_save_path + "signals" + str(power_dBm) + "_cnt.pdf")

        #########
        fig1, (ax) = plt.subplots(nrows=1, figsize=(8, 6))
        for ii, idx in enumerate(coi_list):
            plt.plot(z_max, np.transpose(watt2dBm(
                signal_solution_bi[:, idx])), color="b", linestyle=linestyles[ii], label=labels[ii])

        plt.xlabel("z [km]")
        plt.ylabel("Signal power [dBm]")
        plt.grid("on")
        plt.legend()
        plt.minorticks_on()
        # plt.figure()
        # plt.plot(signal_wavelengths, watt2dBm(signal_solution[-1]), color="k")
        if show_plots:
            plt.show()

        plt.tight_layout()
        plt.savefig(plot_save_path + "signals" + str(power_dBm) + "_bi.pdf")

        ########################################################################
        # ASE ONLY
        ########################################################################
        plt.rcParams['font.size'] = '34'

        fig1, (ax) = plt.subplots(nrows=1, figsize=(8, 8))

        for ii, idx in enumerate(coi_list):
            plt.plot(z_max, np.transpose(watt2dBm(
                ase_solution_co[:, idx])), color="black", linestyle=linestyles[ii], label=labels[ii])
        plt.legend()
        plt.xlabel("z [km]")
        plt.ylabel("ASE power [dBm]")
        plt.grid("on")
        plt.xticks([0, 40, 80], [0, 40,80])
        plt.yticks([-80, -70, -60,  -50, -40], [-80, -70, -60,  -50, -40])
        plt.minorticks_on()
        #plt.annotate("CO", (60, -70))

        if show_plots:
            plt.show()

        plt.tight_layout()
        plt.savefig(plot_save_path + "ases" + str(power_dBm) + "_co.pdf")

        #########
        fig1, (ax) = plt.subplots(nrows=1, figsize=(8, 8))
        for ii, idx in enumerate(coi_list):
            plt.plot(z_max, np.transpose(watt2dBm(
                ase_solution_cnt[:, idx])), color="black", linestyle=linestyles[ii], label=labels[ii])
        plt.legend()   
        plt.xlabel("z [km]")
        plt.ylabel("ASE power [dBm]")
        #plt.annotate("CNT", (60, -70))
        plt.xticks([0, 40, 80], [0, 40,80])
        plt.yticks([-80, -70, -60,  -50, -40], [-80, -70, -60,  -50, -40])
        plt.grid("on")
        plt.minorticks_on()

        if show_plots:
            plt.show()

        plt.tight_layout()
        plt.savefig(plot_save_path + "ases" + str(power_dBm) + "_cnt.pdf")

        #########
        fig1, (ax) = plt.subplots(nrows=1, figsize=(8, 8))
        for ii, idx in enumerate(coi_list):
            plt.plot(z_max, np.transpose(watt2dBm(
                ase_solution_bi[:, idx])), color="black", linestyle=linestyles[ii], label=labels[ii])
        plt.legend()
        #plt.annotate("BI", (60, -70))
        plt.xlabel("z [km]")
        plt.ylabel("ASE power [dBm]")
        plt.xticks([0, 40, 80], [0, 40,80])
        plt.yticks([-80, -70, -60,  -50, -40], [-80, -70, -60,  -50, -40])
        plt.grid("on")
        plt.minorticks_on()
        # plt.figure()
        # plt.plot(ase_wavelengths, watt2dBm(ase_solution[-1]), color="k")
        if show_plots:
            plt.show()

        plt.tight_layout()
        plt.savefig(plot_save_path + "ases" + str(power_dBm) + "_bi.pdf")

        ########################################################################
        # PUMPS ONLY
        ########################################################################
        ######### pumps co
        plt.rcParams['font.size'] = '24'

        # color=cm.viridis(i/(len(m)-10)/3*2)
        fig1, (ax) = plt.subplots(nrows=1, figsize=(8, 6))
        num_CO = len(pump_solution_co[0, :])
        num_CNT = len(pump_solution_cnt[0, :])
        num_bi = len(pump_solution_bi[0, :])

        pump_wavelengths_CO = 1e6*np.load(optimized_result_path+'opt_wavelengths_co' + str(power_dBm) + '.npy')

        for pp in range(num_CO):
            ax_= ax.plot(z_max, watt2dBm(pump_solution_co[:, pp]), color=cm.gist_rainbow(0.95-(0.9*pp/(num_CO))))
        plt.xlabel("z [km]")
        plt.ylabel("Pump power [dBm]")
        plt.grid("on")
        plt.minorticks_on()
        if show_plots:
            plt.show()
        plt.tight_layout()

        #Get cmap values
        cmap_vals = cm.gist_rainbow([item/num_co for item in range(num_CO)])
        
        # Define new cmap
        cmap_new = mpl.colors.LinearSegmentedColormap.from_list('new_cmap', cmap_vals[::-1])
        #norm = mpl.colors.BoundaryNorm(pump_wavelengths_CO, num_CO)
        norm = mpl.colors.Normalize(vmin=np.min(pump_wavelengths_CO), vmax=np.max(pump_wavelengths_CO)) 
        # Define SM
        sm = plt.cm.ScalarMappable(cmap=cmap_new, norm=norm)
        sm.set_array([])

        # Plot colorbar
        clb = plt.colorbar(sm, pad=0.01)
        clb.ax.set_xticks(pump_wavelengths_CO)
        clb.ax.set_xticklabels(["{:1.4f}".format(pmp) for pmp in pump_wavelengths_CO])
        # print(pump_wavelengths_CO)
        clb.set_label(r"Pump wavelenght [$\mu$m]", labelpad=5)

        plt.savefig(plot_save_path + "../pumps" + str(power_dBm) + "_co.pdf")
        
        ######### pumps cnt
        fig1, (ax) = plt.subplots(nrows=1, figsize=(8, 6))
        pump_wavelengths_cnt = 1e6*np.load(optimized_result_path+'opt_wavelengths_cnt' + str(power_dBm) + '.npy')

        for pp in range(num_CNT):
            ax_= ax.plot(z_max, watt2dBm(pump_solution_cnt[:, pp]), color=cm.gist_rainbow(0.95-(0.9*pp/(num_CNT))))
        plt.xlabel("z [km]")
        plt.ylabel("Pump power [dBm]")
        plt.grid("on")
        plt.minorticks_on()
        if show_plots:
            plt.show()
        plt.tight_layout()

        #Get cmap values
        cmap_vals = cm.gist_rainbow([item/num_CNT for item in range(num_CNT)])
        
        # Define new cmap
        cmap_new = mpl.colors.LinearSegmentedColormap.from_list('new_cmap', cmap_vals[::-1])
        #norm = mpl.colors.BoundaryNorm(pump_wavelengths_cnt, num_CNT)
        norm = mpl.colors.Normalize(vmin=np.min(pump_wavelengths_cnt), vmax=np.max(pump_wavelengths_cnt)) 
        # Define SM
        sm = plt.cm.ScalarMappable(cmap=cmap_new, norm=norm)
        sm.set_array([])

        # Plot colorbar
        clb = plt.colorbar(sm, pad=0.01)
        clb.ax.set_xticks(pump_wavelengths_cnt)
        clb.ax.set_xticklabels(["{:1.4f}".format(pmp) for pmp in pump_wavelengths_cnt])
        # print(pump_wavelengths_cnt)
        clb.set_label(r"Pump wavelenght [$\mu$m]", labelpad=5)

        plt.savefig(plot_save_path + "../pumps" + str(power_dBm) + "_cnt.pdf")
        
        ######### pumps bi
        #
        fig1, (ax) = plt.subplots(nrows=1, figsize=(8, 6))
        pump_wavelengths_bi = 1e6*np.load(optimized_result_path_bi+'opt_wavelengths_bi' + str(power_dBm) + '.npy')

        for pp in range(num_bi):
            ax_= ax.plot(z_max, watt2dBm(pump_solution_bi[:, pp]), color=cm.gist_rainbow(0.95-(0.9*pp/(num_bi))))
        plt.xlabel("z [km]")
        plt.ylabel("Pump power [dBm]")
        plt.grid("on")
        plt.minorticks_on()
        if show_plots:
            plt.show()
        plt.tight_layout()

        #Get cmap values
        cmap_vals = cm.gist_rainbow([item/num_bi for item in range(num_bi)])
        
        # Define new cmap
        cmap_new = mpl.colors.LinearSegmentedColormap.from_list('new_cmap', cmap_vals[::-1])
        #norm = mpl.colors.BoundaryNorm(pump_wavelengths_bi, num_bi)
        norm = mpl.colors.Normalize(vmin=np.min(pump_wavelengths_bi), vmax=np.max(pump_wavelengths_bi)) 
        # Define SM
        sm = plt.cm.ScalarMappable(cmap=cmap_new, norm=norm)
        sm.set_array([])

        # Plot colorbar
        clb = plt.colorbar(sm, pad=0.01)
        clb.ax.set_xticks(pump_wavelengths_bi)
        clb.ax.set_xticklabels(["{:1.4f}".format(pmp) for pmp in pump_wavelengths_bi])
        # print(pump_wavelengths_bi)
        clb.set_label(r"Pump wavelenght [$\mu$m]", labelpad=5)

        plt.savefig(plot_save_path + "pumps" + str(power_dBm) + "_bi.pdf")
        
        # pump wavelengths and power
        #################################
        ## the pumps are allright
        #################################
        fig1, (ax) = plt.subplots(nrows=1, figsize=(8, 6))
        pump_wavelengths_bi = 1e6*np.load(optimized_result_path_bi+'opt_wavelengths_bi' + str(power_dBm) + '.npy')
        pump_powers_bi = np.load(optimized_result_path_bi+'opt_powers_bi' + str(power_dBm) + '.npy')
        print(pump_powers_bi)

        cmap_new = mpl.colors.LinearSegmentedColormap.from_list('new_cmap', cmap_vals[::-1])
        #Get cmap values

        markerline, stemline, baseline, = plt.stem(1e6*pump_wavelengths_bi, 10*np.log10(pump_powers_bi)+30, linefmt=None, markerfmt="x", basefmt=None)
        plt.setp(stemline, linewidth = 3)
        plt.setp(markerline, markersize = 15)
        cmap_vals = cm.gist_rainbow([item/num_bi for item in range(num_bi)])
        
        # Define new cmap

        #norm = mpl.colors.BoundaryNorm(pump_wavelengths_bi, num_bi)
        norm = mpl.colors.Normalize(vmin=np.min(pump_wavelengths_bi), vmax=np.max(pump_wavelengths_bi)) 
        # Define SM
        sm = plt.cm.ScalarMappable(cmap=cmap_new, norm=norm)
        sm.set_array([])

        # Plot colorbar
        # clb = plt.colorbar(sm, pad=0.01)
        # clb.ax.set_xticks(pump_wavelengths_bi)
        # clb.ax.set_xticklabels(["{:1.4f}".format(pmp) for pmp in pump_wavelengths_bi])
        # # print(pump_wavelengths_bi)
        plt.xlabel(r"Pump wavelength [$\mu$m]")
        plt.ylabel("Pump power [dBm]")
        plt.grid("on")
        plt.minorticks_on()
        plt.tight_layout()

        plt.savefig(plot_save_path + "stems" + str(power_dBm) + "_bi.pdf")

        #################################
        ## the signals are allright
        #################################
        fig1, (ax) = plt.subplots(nrows=1, figsize=(8, 6))

        print(signal_solution_bi[-1, :])
        plt.plot(1e6*signal_wavelengths, 10*np.log10(signal_solution_bi[-1, ::-1])-10*np.log10(signal_solution_bi[0, :]), marker="x", markersize=10)
        #Get cmap values
        cmap_vals = cm.gist_rainbow([item/num_bi for item in range(num_bi)])
        
        # Define new cmap

        #norm = mpl.colors.BoundaryNorm(pump_wavelengths_bi, num_bi)
        norm = mpl.colors.Normalize(vmin=np.min(pump_wavelengths_bi), vmax=np.max(pump_wavelengths_bi)) 
        # Define SM
        sm = plt.cm.ScalarMappable(cmap=cmap_new, norm=norm)
        sm.set_array([])
        plt.xlabel(r"Signal wavelength [$\mu$m]")
        plt.ylabel("Signal gain [dB]")
        # Plot colorbar
        # clb = plt.colorbar(sm, pad=0.01)
        # clb.ax.set_xticks(pump_wavelengths_bi)
        # clb.ax.set_xticklabels(["{:1.4f}".format(pmp) for pmp in pump_wavelengths_bi])
        # # print(pump_wavelengths_bi)
        # clb.set_label(r"Pump wavelenght [$\mu$m]", labelpad=5)
        ax.axhline(-3, color="red", linestyle="dotted", linewidth=2)

        plt.grid()
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig(plot_save_path + "oo_gain" + str(power_dBm) + "_bi.pdf")