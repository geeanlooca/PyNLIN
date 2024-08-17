import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
from pynlin.utils import dBm2watt, watt2dBm
import pynlin.constellations
from matplotlib.lines import Line2D
import matplotlib as mpl
import json

f = open("./scripts/sim_config.json")
data = json.load(f)
dispersion = data["dispersion"]
effective_area = data["effective_area"]
baud_rate = data["baud_rate"]
fiber_lengths = data["fiber_length"]
num_channels = data["num_channels"]
channel_spacing = data["channel_spacing"]
center_frequency = data["center_frequency"]
store_true = data["store_true"]
pulse_shape = data["pulse_shape"]
partial_collision_margin = data["partial_collision_margin"]
num_co = data["num_co"]
num_ct = data["num_ct"]
wavelength = data["wavelength"]
special = data["special"]
pump_direction = data["pump_direction"]
num_only_co_pumps = data['num_only_co_pumps']
num_only_ct_pumps = data['num_only_ct_pumps']
"z [km]"
plot_width = 10
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '500'
plt.rcParams['font.size'] = '24'
for fiber_length in fiber_lengths:
    length_setup = int(fiber_length * 1e-3)
    plot_save_path = "/home/lorenzi/Scrivania/progetti/NLIN/plots_" + \
        str(length_setup) + '/' + str(num_co) + '_co_' + \
        str(num_ct) + '_ct_' + special + '/profiles/'
    #
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    #
    results_path_co = '../results_' + \
        str(length_setup) + '/' + str(num_only_co_pumps) + '_co/'
    results_path_ct = '../results_' + \
        str(length_setup) + '/' + str(num_only_ct_pumps) + '_ct/'
    results_path_bi = '../results_' + \
        str(length_setup) + '/' + str(num_co) + \
        '_co_' + str(num_ct) + '_ct_' + special + '/'
    #
    optimized_result_path_co = '../results_' + \
        str(length_setup) + '/optimization/' + str(num_only_co_pumps) + '_co/'
    optimized_result_path_ct = '../results_' + \
        str(length_setup) + '/optimization/' + str(num_only_ct_pumps) + '_ct/'
    optimized_result_path_bi = '../results_' + \
        str(length_setup) + '/optimization/' + str(num_co) + \
        '_co_' + str(num_ct) + '_ct_' + special + '/'
    time_integrals_results_path = '../results/'

    # PLOTTING PARAMETERS
    interfering_grid_index = 1
    # power_dBm_list = [-20, -10, -5, 0]
    power_dBm_list = [0.0, -10.0, -20.0]
    power_dBm_list = [-20.0, -10.0, 0.0]

    arity_list = [16]

    wavelength = 1550
    baud_rate = 10 * 1e9
    dispersion = 18
    channel_spacing = 100
    num_channels = 50

    beta2 = pynlin.utils.dispersion_to_beta2(
        dispersion, wavelength
    )
    fiber = pynlin.fiber.Fiber(
        effective_area=80e-12,
        beta2=beta2
    )
    wdm = pynlin.wdm.WDM(
        spacing=channel_spacing,
        num_channels=num_channels,
        center_frequency=190e12
    )
    signal_wavelengths = wdm.wavelength_grid()

    partial_collision_margin = 5
    points_per_collision = 10

    print("beta2: ", fiber.beta2)
    print("gamma: ", fiber.gamma)

    Delta_theta_2_co = np.zeros_like(
        np.ndarray(shape=(len(power_dBm_list), len(arity_list)))
    )
    Delta_theta_2_ct = np.zeros_like(Delta_theta_2_co)

    show_plots = False
    show_pumps = False
    linestyles = ["solid", "dashed", "dotted"]
    coi_list = [0, 24, 49]
    # if input("\nASE-Signal_pump profile plotter: \n\t>Length= " + str(length_setup) + "km \n\t>power list= " + str(power_dBm_list) + "\nAre you sure? (y/[n])") != "y":
    #   exit()
    gain_dB = 0.0
    configs = [[num_co, num_ct]]
    for config in configs:
        num_co = config[0]
        num_ct = config[1]
        for idx, power_dBm in enumerate(power_dBm_list):
            average_power = dBm2watt(power_dBm)
            # SIMULATION DATA LOAD =================================

            pump_solution_co = np.load(
                results_path_co + 'pump_solution_co_' + str(power_dBm) + '.npy')
            signal_solution_co = np.load(
                results_path_co + 'signal_solution_co_' + str(power_dBm) + '.npy')
            ase_solution_co = np.load(
                results_path_co + 'ase_solution_co_' + str(power_dBm) + '.npy')

            pump_solution_ct = np.load(
                results_path_ct + 'pump_solution_ct_' + str(power_dBm) + '.npy')
            signal_solution_ct = np.load(
                results_path_ct + 'signal_solution_ct_' + str(power_dBm) + "_opt_gain_" + str(gain_dB) + '.npy')

            ase_solution_ct = np.load(
                results_path_ct + 'ase_solution_ct_' + str(power_dBm) + '.npy')

            pump_solution_bi = np.load(
                results_path_bi + 'pump_solution_bi_' + str(power_dBm) + '.npy')
            signal_solution_bi = np.load(
                results_path_bi + 'signal_solution_bi_' + str(power_dBm) + '.npy')
            ase_solution_bi = np.load(
                results_path_bi + 'ase_solution_bi_' + str(power_dBm) + '.npy')
            # plt.figure()
            # plt.plot(signal_wavelengths, watt2dBm(signal_solution_co[-1]), color="k")
            # plt.figure()
            # plt.plot(signal_wavelengths, watt2dBm(signal_solution_ct[-1]), color="k")
            # plt.figure()
            # plt.plot(signal_wavelengths, watt2dBm(signal_solution_bi[-1]), color="k")

            # z_max = np.load(results_path + 'z_max.npy')
            # f = h5py.File(results_path + '0_9_results.h5', 'r')
            z_max = np.linspace(0, fiber_length, np.shape(pump_solution_bi)[0])
            z_max *= 1e-3
            custom_lines = [Line2D([0], [0], color="k", lw=2),
                            Line2D([0], [0], color="b", lw=2)]
            if show_pumps:
                custom_lines.append(Line2D([0], [0], color="r", lw=2))

            ########################################################################
            # SIGNALS AND ASE ONLY
            ########################################################################
            if False:
                fig1, (ax) = plt.subplots(nrows=1, figsize=(plot_width, 6))

                plt.plot(z_max, np.transpose(
                    watt2dBm([ase_solution_co[:, idx] for idx in [0, 24, 49]])), color="k")
                plt.plot(z_max, np.transpose(
                    watt2dBm([signal_solution_co[:, idx] for idx in [0, 24, 49]])), color="b")
                if show_pumps:
                    plt.plot(z_max, watt2dBm(pump_solution_co), color="r")

                plt.legend(custom_lines, ['ASE', 'Signal', 'Pump'])

                plt.xlabel(r"$z$ [km]")
                plt.ylabel("Wave power [dBm]")
                plt.grid("on")
                plt.minorticks_on()
                osnr = 10 * \
                    np.log10(signal_solution_co[-1:, :] / ase_solution_co[-1:, :])
                # plt.figure()
                # plt.plot(signal_wavelengths, watt2dBm(signal_solution[-1]), color="k")
                if show_plots:
                    plt.show()

                plt.tight_layout()
                plt.savefig(plot_save_path + "co_profile" + str(power_dBm) + ".pdf")

                ###############
                fig1, (ax) = plt.subplots(nrows=1, figsize=(plot_width, 6))
                plt.plot(z_max, np.transpose(
                    watt2dBm([ase_solution_ct[:, idx] for idx in [0, 24, 49]])), color="k")
                plt.plot(z_max, np.transpose(
                    watt2dBm([signal_solution_ct[:, idx] for idx in [0, 24, 49]])), color="b")
                if show_pumps:
                    plt.plot(z_max, watt2dBm(pump_solution_ct), color="r")

                plt.legend(custom_lines, ['ASE', 'Signal', 'Pump'])

                plt.xlabel(r"$z$ [km]")
                plt.ylabel("Wave power [dBm]")
                plt.grid("on")
                plt.minorticks_on()
                osnr = 10 * \
                    np.log10(signal_solution_ct[-1:, :] / ase_solution_ct[-1:, :])
                # plt.figure()
                # plt.plot(signal_wavelengths, watt2dBm(signal_solution[-1]), color="k")
                if show_plots:
                    plt.show()

                plt.tight_layout()
                plt.savefig(plot_save_path + "ct_profile" + str(power_dBm) + ".pdf")

                #########
                fig1, (ax) = plt.subplots(nrows=1, figsize=(plot_width, 6))
                plt.plot(z_max, np.transpose(
                    watt2dBm([ase_solution_bi[:, idx] for idx in [0, 24, 49]])), color="k")
                plt.plot(z_max, np.transpose(
                    watt2dBm([signal_solution_bi[:, idx] for idx in [0, 24, 49]])), color="b")
                if show_pumps:
                    plt.plot(z_max, watt2dBm(pump_solution_bi), color="r")

                plt.legend(custom_lines, ['ASE', 'Signal', 'Pump'])

                plt.xlabel(r"$z$ [km]")
                plt.ylabel("Wave power [dBm]")
                plt.grid("on")
                plt.minorticks_on()
                osnr = 10 * \
                    np.log10(signal_solution_bi[-1:, :] / ase_solution_bi[-1:, :])
                # plt.figure()
                # plt.plot(signal_wavelengths, watt2dBm(signal_solution[-1]), color="k")
                if show_plots:
                    plt.show()

                plt.tight_layout()
                plt.savefig(plot_save_path + "bi_profile" + str(power_dBm) + ".pdf")

            ########################################################################
            # SIGNALS ONLY
            ########################################################################
            labels = ["ch.1", "ch.25", "ch.50"]
            fig1, (ax) = plt.subplots(nrows=1, figsize=(plot_width, 6))
            for ii, idx in enumerate(coi_list):
                plt.plot(z_max, np.transpose(watt2dBm(
                    signal_solution_co[:, idx])), color="b", linestyle=linestyles[ii], label=labels[ii])

            plt.xlabel(r"$z$ [km]")
            plt.ylabel("Signal power [dBm]")
            plt.grid("on")
            plt.legend()
            plt.minorticks_on()

            if show_plots:
                plt.show()

            plt.tight_layout()
            plt.savefig(plot_save_path + "co_signals" + str(power_dBm) + ".pdf")

            fig1, (ax) = plt.subplots(nrows=1, figsize=(plot_width, 6))
            for ii, idx in enumerate(coi_list):
                plt.plot(z_max, np.transpose(watt2dBm(
                    signal_solution_ct[:, idx])), color="b", linestyle=linestyles[ii], label=labels[ii])

            plt.xlabel(r"$z$ [km]")
            plt.ylabel("Signal power [dBm]")
            plt.grid("on")
            plt.legend()

            plt.minorticks_on()

            if show_plots:
                plt.show()

            plt.tight_layout()
            plt.savefig(plot_save_path + "ct_signals" +
                        str(power_dBm) + "_gain_" + str(gain_dB) + ".pdf")

            #########
            fig1, (ax) = plt.subplots(nrows=1, figsize=(plot_width, 6))
            for ii, idx in enumerate(coi_list):
                plt.plot(z_max, np.transpose(watt2dBm(
                    signal_solution_bi[:, idx])), color="b", linestyle=linestyles[ii], label=labels[ii])

            plt.xlabel(r"$z$ [km]")
            plt.ylabel("Signal power [dBm]")
            plt.grid("on")
            plt.legend()
            plt.minorticks_on()
            # plt.figure()
            # plt.plot(signal_wavelengths, watt2dBm(signal_solution[-1]), color="k")
            if show_plots:
                plt.show()

            plt.tight_layout()
            plt.savefig(plot_save_path + "bi_signals" + str(power_dBm) + ".pdf")

            ########################################################################
            # ASE ONLY
            ########################################################################
            plt.rcParams['font.size'] = '34'

            fig1, (ax) = plt.subplots(nrows=1, figsize=(plot_width, 6))

            for ii, idx in enumerate(coi_list):
                plt.plot(z_max, np.transpose(watt2dBm(
                    ase_solution_co[:, idx])), color="black", linestyle=linestyles[ii], label=labels[ii])
            plt.legend()
            plt.xlabel(r"$z$ [km]")
            plt.ylabel("ASE power [dBm]")
            plt.grid("on")
            plt.xticks([0, 40, 80], [0, 40, 80])
            plt.yticks([-80, -70, -60, -50, -40], [-80, -70, -60, -50, -40])
            plt.minorticks_on()
            # plt.annotate("CO", (60, -70))

            if show_plots:
                plt.show()

            plt.tight_layout()
            plt.savefig(plot_save_path + "co_ases" + str(power_dBm) + ".pdf")

            #########
            fig1, (ax) = plt.subplots(nrows=1, figsize=(plot_width, 6))
            for ii, idx in enumerate(coi_list):
                plt.plot(z_max, np.transpose(watt2dBm(
                    ase_solution_ct[:, idx])), color="black", linestyle=linestyles[ii], label=labels[ii])
            plt.legend()
            plt.xlabel(r"$z$ [km]")
            plt.ylabel("ASE power [dBm]")
            # plt.annotate("ct", (60, -70))
            plt.xticks([0, 40, 80], [0, 40, 80])
            plt.yticks([-80, -70, -60, -50, -40], [-80, -70, -60, -50, -40])
            plt.grid("on")
            plt.minorticks_on()

            if show_plots:
                plt.show()

            plt.tight_layout()
            plt.savefig(plot_save_path + "ct_ases" + str(power_dBm) + ".pdf")

            #########
            fig1, (ax) = plt.subplots(nrows=1, figsize=(plot_width, 6))
            for ii, idx in enumerate(coi_list):
                plt.plot(z_max, np.transpose(watt2dBm(
                    ase_solution_bi[:, idx])), color="black", linestyle=linestyles[ii], label=labels[ii])
            plt.legend()
            # plt.annotate("BI", (60, -70))
            plt.xlabel(r"$z$ [km]")
            plt.ylabel("ASE power [dBm]")
            plt.xticks([0, 40, 80], [0, 40, 80])
            plt.yticks([-80, -70, -60, -50, -40], [-80, -70, -60, -50, -40])
            plt.grid("on")
            plt.minorticks_on()
            # plt.figure()
            # plt.plot(ase_wavelengths, watt2dBm(ase_solution[-1]), color="k")
            if show_plots:
                plt.show()

            plt.tight_layout()
            plt.savefig(plot_save_path + "bi_ases" + str(power_dBm) + ".pdf")

            ########################################################################
            # PUMPS ONLY
            ########################################################################

            # pumps ct
            fig1, (ax) = plt.subplots(nrows=1, figsize=(plot_width, 6))
            pump_wavelengths_ct = 1e6 * \
                np.load(optimized_result_path_ct + 'opt_wavelengths_ct' +
                        str(power_dBm) + "_gain_" + str(gain_dB) + '.npy')

            lambda_max = np.max(pump_wavelengths_ct)
            lambda_min = np.min(pump_wavelengths_ct)
            cmap = cm.rainbow([(0.9 * (item - lambda_min) / (lambda_max -
                              lambda_min)) - 0.05 for item in pump_wavelengths_ct])

            for pp in range(num_ct):
                ax_ = ax.plot(z_max, watt2dBm(pump_solution_ct[:, pp]), color=cmap[pp])
            plt.xlabel(r"$z$ [km]")
            plt.ylabel("Pump power [dBm]")
            plt.grid("on")
            plt.minorticks_on()
            if show_plots:
                plt.show()
            plt.tight_layout()

            # Get cmap values
            cmap_vals = cm.rainbow([item / num_ct for item in range(num_ct)])

            # Define new cmap
            norm = mpl.colors.Normalize(vmin=lambda_min, vmax=lambda_max)
            # Define SM
            sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=norm)
            sm.set_array([])

            # Plot colorbar
            clb = plt.colorbar(sm, pad=0.01)
            clb.ax.set_yticks(pump_wavelengths_ct)
            clb.ax.set_yticklabels(["{:1.3f}".format(pmp)
                                   for pmp in pump_wavelengths_ct])
            # print(pump_wavelengths_ct)
            clb.set_label(r"Pump wavelenght [$\mu$m]", labelpad=5)

            plt.savefig(plot_save_path + "ct_pumps" + str(power_dBm) +
                        "_gain_" + str(gain_dB) + ".pdf")

            # pumps co
            plt.rcParams['font.size'] = '24'

            # color=cm.viridis(i/(len(m)-10)/3*2)
            fig1, (ax) = plt.subplots(nrows=1, figsize=(plot_width, 6))
            num_CO = len(pump_solution_co[0, :])
            num_ct = len(pump_solution_ct[0, :])
            num_bi = len(pump_solution_bi[0, :])

            pump_wavelengths_co = 1e6 * \
                np.load(optimized_result_path_co +
                        'opt_wavelengths_co' + str(power_dBm) + '.npy')

            lambda_max = np.max(pump_wavelengths_co)
            lambda_min = np.min(pump_wavelengths_co)
            cmap = cm.rainbow([(0.9 * (item - lambda_min) / (lambda_max -
                              lambda_min)) - 0.05 for item in pump_wavelengths_co])

            for pp in range(num_CO):
                ax_ = ax.plot(z_max, watt2dBm(pump_solution_co[:, pp]), color=cmap[pp])
            plt.xlabel(r"$z$ [km]")
            plt.ylabel("Pump power [dBm]")
            plt.grid("on")
            plt.minorticks_on()
            if show_plots:
                plt.show()
            plt.tight_layout()

            # Get cmap values
            # norm = mpl.colors.BoundaryNorm(pump_wavelengths_co, num_CO)

            norm = mpl.colors.Normalize(vmin=lambda_min, vmax=lambda_max)
            # Define SM
            sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=norm)
            sm.set_array([])

            # Plot colorbar

            clb = plt.colorbar(sm, pad=0.01)
            clb.ax.set_yticks(pump_wavelengths_co)
            clb.ax.set_yticklabels(["{:1.3f}".format(pmp)
                                   for pmp in pump_wavelengths_co])
            clb.set_label(r"Pump wavelenght [$\mu$m]", labelpad=5)

            plt.savefig(plot_save_path + "co_pumps" + str(power_dBm) + ".pdf")

            # pumps bi
            fig1, (ax) = plt.subplots(nrows=1, figsize=(plot_width, 6))
            pump_wavelengths_bi = 1e6 * \
                np.load(optimized_result_path_bi +
                        'opt_wavelengths_bi' + str(power_dBm) + '.npy')

            lambda_max = np.max(pump_wavelengths_bi)
            lambda_min = np.min(pump_wavelengths_bi)
            cmap = cm.rainbow([(0.9 * (item - lambda_min) / (lambda_max -
                              lambda_min)) - 0.05 for item in pump_wavelengths_bi])

            for pp in range(num_bi):
                ax_ = ax.plot(z_max, watt2dBm(pump_solution_bi[:, pp]), color=cmap[pp])
            plt.xlabel(r"$z$ [km]")
            plt.ylabel("Pump power [dBm]")
            plt.grid("on")
            plt.minorticks_on()
            if show_plots:
                plt.show()
            plt.tight_layout()

            # Get cmap values

            norm = mpl.colors.Normalize(vmin=lambda_min, vmax=lambda_max)
            # Define SM
            sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=norm)
            sm.set_array([])

            # Plot colorbar
            clb = plt.colorbar(sm, pad=0.01)
            clb.ax.set_yticks(pump_wavelengths_bi)
            clb.ax.set_yticklabels(["{:1.3f}".format(pmp)
                                   for pmp in pump_wavelengths_bi])
            clb.set_label(r"Pump wavelenght [$\mu$m]", labelpad=5)

            plt.savefig(plot_save_path + "bi_pumps" + str(power_dBm) + ".pdf")

            # pump wavelengths and power

            fig1, (ax) = plt.subplots(nrows=1, figsize=(plot_width, 6))
            pump_wavelengths_bi = 1e6 * \
                np.load(optimized_result_path_bi +
                        'opt_wavelengths_bi' + str(power_dBm) + '.npy')
            pump_powers_bi = np.load(optimized_result_path_bi +
                                     'opt_powers_bi' + str(power_dBm) + '.npy')
            print("\n\nwlngths bi: ", pump_wavelengths_bi)
            print("pwrs bi: ", watt2dBm(pump_powers_bi))
            print("front powers bi: ", watt2dBm(pump_solution_bi[0, :]))
            print("end powers bi: ", watt2dBm(pump_solution_bi[-1, :]))

            cmap_new = mpl.colors.LinearSegmentedColormap.from_list(
                'new_cmap', cmap_vals[::-1])

            if pump_direction != "direct":
                print("\nWarning: stem plot do not respect real pump directions!")
            markerline, stemline, baseline, = plt.stem(1e6 * pump_wavelengths_bi[:num_co], 10 * np.log10(
                pump_powers_bi[:num_co]) + 30, linefmt=None, markerfmt="x", basefmt=None)
            plt.setp(stemline, linewidth=3, label="co")
            plt.setp(markerline, markersize=15)
            markerline, stemline, baseline, = plt.stem(1e6 * pump_wavelengths_bi[num_co:], 10 * np.log10(
                pump_powers_bi[num_co:]) + 30, linefmt=None, markerfmt="o", basefmt=None)
            plt.setp(stemline, linewidth=3, label="ct")
            plt.setp(markerline, markersize=15)
            cmap_vals = cm.rainbow([item / num_bi for item in range(num_bi)])
            # Define new cmap

            norm = mpl.colors.Normalize(vmin=np.min(
                pump_wavelengths_bi), vmax=np.max(pump_wavelengths_bi))
            # Define SM
            sm = plt.cm.ScalarMappable(cmap=cmap_new, norm=norm)
            sm.set_array([])
            plt.legend()

            plt.xlabel(r"Pump wavelength [$\mu$m]")
            plt.ylabel("Pump power [dBm]")
            plt.grid("on")
            plt.minorticks_on()
            plt.tight_layout()

            plt.savefig(plot_save_path + "bi_stems" + str(power_dBm) + "_bi.pdf")

            # COPUMPS
            fig1, (ax) = plt.subplots(nrows=1, figsize=(plot_width, 6))
            pump_wavelengths_co = 1e6 * \
                np.load(optimized_result_path_co +
                        'opt_wavelengths_co' + str(power_dBm) + '.npy')
            pump_powers_co = np.load(optimized_result_path_co +
                                     'opt_powers_co' + str(power_dBm) + '.npy')
            print("\n\nwlngths co: ", pump_wavelengths_co)
            print("pwrs co: ", watt2dBm(pump_powers_co))

            cmap_new = mpl.colors.LinearSegmentedColormap.from_list(
                'new_cmap', cmap_vals[::-1])

            if pump_direction != "direct":
                print("\nWarning: stem plot do not respect real pump directions!")
            markerline, stemline, baseline, = plt.stem(
                1e6 * pump_wavelengths_co, 10 * np.log10(pump_powers_co) + 30, linefmt=None, markerfmt="x", basefmt=None)
            plt.setp(stemline, linewidth=3, label="co")
            plt.setp(markerline, markersize=15)
            cmap_vals = cm.rainbow([item / num_co for item in range(num_co)])
            plt.legend()
            # Define new cmap
            norm = mpl.colors.Normalize(vmin=np.min(
                pump_wavelengths_co), vmax=np.max(pump_wavelengths_co))
            # Define SM
            sm = plt.cm.ScalarMappable(cmap=cmap_new, norm=norm)
            sm.set_array([])

            plt.xlabel(r"Pump wavelength [$\mu$m]")
            plt.ylabel("Pump power [dBm]")
            plt.grid("on")
            plt.minorticks_on()
            plt.tight_layout()

            plt.savefig(plot_save_path + "co_stems" + str(power_dBm) + ".pdf")

            pump_wavelengths_ct = 1e6 * \
                np.load(optimized_result_path_ct +
                        'opt_wavelengths_ct' + str(power_dBm) + '.npy')
            pump_powers_ct = np.load(optimized_result_path_ct +
                                     'opt_powers_ct' + str(power_dBm) + '.npy')
            print("\n\nwlngths ct: ", pump_wavelengths_ct)
            print("pwrs ct: ", watt2dBm(pump_powers_ct))
            print("end powers bi: ", watt2dBm(pump_solution_ct[-1, :]))

            # Signals

            fig1, (ax) = plt.subplots(nrows=1, figsize=(plot_width, 6))
            plt.plot(1e6 * signal_wavelengths, 10 * np.log10(signal_solution_bi[-1, ::-1]) - 10 * np.log10(
                signal_solution_bi[0, :]), marker="x", markersize=10)
            # Get cmap values
            cmap_vals = cm.rainbow([item / num_bi for item in range(num_bi)])

            # Define new cmap
            norm = mpl.colors.Normalize(vmin=np.min(
                pump_wavelengths_bi), vmax=np.max(pump_wavelengths_bi))
            # Define SM
            sm = plt.cm.ScalarMappable(cmap=cmap_new, norm=norm)
            sm.set_array([])
            plt.xlabel(r"Signal wavelength [$\mu$m]")
            plt.ylabel("Signal gain [dB]")
            ax.axhline(-3, color="red", linestyle="dotted", linewidth=2)

            plt.grid()
            plt.minorticks_on()
            plt.tight_layout()
            plt.savefig(plot_save_path + "oo_gain" + str(power_dBm) + "_bi.pdf")
