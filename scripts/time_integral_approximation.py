import argparse
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "-R", "--baud-rate", default=10, help="The baud rate of each WDM channel in GHz."
)
parser.add_argument(
    "-D",
    "--dispersion",
    default=18,
    type=float,
    help="The dispersion coefficient of the fiber in ps/nm km.",
)
parser.add_argument(
    "-L",
    "--fiber-length",
    default=100,
    type=float,
    help="The length of the fiber in kilometers.",
)
parser.add_argument(
    "-c",
    "--channel-spacing",
    default=100,
    type=float,
    help="The spacing between neighboring WDM channels in GHz.",
)
parser.add_argument(
    "-C",
    "--channel-count",
    default=50,
    type=int,
    help="The number of WDM channels in the grid.",
)
parser.add_argument(
    "-M",
    "--use-multiprocessing",
    action="store_true",
    help="If passed, this flag enables multicore processing to compute the collisions in parallel.",
)
parser.add_argument(
    "-W",
    "--wavelength",
    default=1550,
    type=float,
    help="The wavelength at which the dispersion coefficient is given (in nanometers).",
)

args = parser.parse_args()

print(args)
beta2 = pynlin.utils.dispersion_to_beta2(
    args.dispersion * 1e-12 / (1e-9 * 1e3), args.wavelength * 1e-9
)
fiber_length = args.fiber_length * 1e3
channel_spacing = args.channel_spacing * 1e9
num_channels = args.channel_count
baud_rate = args.baud_rate * 1e9

fiber = pynlin.fiber.Fiber(effective_area=80e-12, beta2=beta2)
wdm = pynlin.wdm.WDM(
    spacing=args.channel_spacing, num_channels=num_channels, center_frequency=190
)

# compute the collisions between the two furthest WDM channels
frequency_of_interest = wdm.frequency_grid()[0]
interfering_frequency = wdm.frequency_grid()[1]
channel_spacing = interfering_frequency - frequency_of_interest


least_steps = 5
most_steps = 50
increment = 5
integration_steps = range(least_steps, most_steps, increment)
integrals = []
pbar_description = (
        f"Test integration with steps from {least_steps} to {most_steps}, total tests {np.ceil((most_steps-least_steps)/increment) }"
    )
collisions_pbar = tqdm.tqdm(integration_steps, leave=False)
collisions_pbar.set_description(pbar_description)

for steps in collisions_pbar:
    z, I, m = pynlin.nlin.compute_all_collisions_X0mm_time_integrals(
        frequency_of_interest,
        interfering_frequency,
        baud_rate,
        fiber,
        fiber_length,
        pulse_shape = "Gaussian",
        rolloff_factor=0.1,
        samples_per_symbol=steps,
        points_per_collision=100,
        use_multiprocessing=True,
        partial_collisions_start=1,
        partial_collisions_end=1,
    )
    integrals.append(I)

# use the approximation given in the paper ~= 1/beta2 * Omega and compare it to the numerical results
stacked_ints = np.stack(integrals)
approx_constant =- 1 / (beta2 * 2 * np.pi * channel_spacing)
approximation = np.ones_like(m, dtype=float)
approximation *= approx_constant

# compute the X0mm coefficients given the precompute time integrals:
# bonus of this approach -> no need to recompute the time integrals if
# we want to compare different amplification schemes or constellations
locs = pynlin.nlin.get_collision_location(m, fiber, channel_spacing, 1 / baud_rate)

# plot the integral convergence with various time integration steps
colormap = plt.cm.gist_ncar 
colors = [colormap(i) for i in np.linspace(0, 1,len(locs))]

plt.figure(figsize=(10, 5))
labels = []
for l, loc in enumerate(locs):
    error = np.subtract(np.abs(pynlin.nlin.Xhkm_precomputed(z, stacked_ints[:, l, :], amplification_function=None)), approx_constant)/approx_constant
    plt.plot(integration_steps, error, marker="+", color=colors[l])
    labels.append(r'collision %i' % (l))
plt.title("difference between computation and approximation, for all collision indexes vs step number")
plt.xlabel("Number of samples per integral")
plt.ylabel("relative approximation error")
plt.legend(labels, ncol=5, loc='upper right', 
           bbox_to_anchor=[0.5, 1.1], 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
plt.show()

# plot X0mm with least accurate step and most accurate step
X0mm_first = pynlin.nlin.Xhkm_precomputed(z, integrals[0], amplification_function=None)
X0mm_last = pynlin.nlin.Xhkm_precomputed(z, integrals[len(integrals)-1], amplification_function=None)
plt.figure()
plt.semilogy(m, np.abs(X0mm_first), marker="o", label="Lowest sampling")
plt.semilogy(m, np.abs(X0mm_last), marker="o", label="Highest sampling")
plt.semilogy(m, np.abs(approximation), marker="x", label=r"$1/(\beta_2 \Omega)$")
plt.minorticks_on()
plt.grid(which="both")
plt.xlabel(r"Collision index $m$")
plt.ylabel(r"$X_{0,m,m}$")
plt.title(
    rf"$f_B(z)=1$, $D={args.dispersion}$ ps/(nm km), $L={args.fiber_length}$ km, $R={args.baud_rate}$ GHz"
)
plt.legend()

# plot absolute value of X0mm in linear scale
plt.figure()
plt.plot(m, np.abs(X0mm_first), marker="o", label="Lowest sampling")
plt.plot(m, np.abs(X0mm_last), marker="o", label="Highest sampling")
plt.plot(m, np.abs(approximation), marker="x", label=r"$1/(\beta_2 \Omega)$")
plt.minorticks_on()
plt.grid(which="both")
plt.xlabel(r"Collision index $m$")
plt.ylabel(r"$X_{0,m,m}$")
plt.title(
    rf"$f_B(z)=1$, $D={args.dispersion}$ ps/(nm km), $L={args.fiber_length}$ km, $R={args.baud_rate}$ GHz"
)
plt.legend()

plt.show()
