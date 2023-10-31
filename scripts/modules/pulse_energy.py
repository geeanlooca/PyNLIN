from argparse import ArgumentParser
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.signal
import pynlin.constellations
import pynlin.pulses

parser = ArgumentParser()
parser.add_argument("-r", "--rolloff", default=0.1, type=float)
parser.add_argument("-n", "--train-length", default=1000, type=int)
args = parser.parse_args()

baud_rate = 1e9
samples_per_symbol = 20
filter_num_symbols = 100
dt = 1 / baud_rate / samples_per_symbol


def generate_symbol_sequence(constellation_symbols, seq_length):
    symbols = random.choices(constellation_symbols, k=seq_length)
    return np.array(symbols)


def generate_train(symbols_seq, samples_per_symbol):
    seq = np.zeros((len(symbols_seq) * samples_per_symbol,), dtype=complex)
    seq[::samples_per_symbol] = symbols_seq
    return seq


pulse = pynlin.pulses.RaisedCosinePulse(
    baud_rate=baud_rate,
    samples_per_symbol=samples_per_symbol,
    num_symbols=filter_num_symbols,
    rolloff=args.rolloff,
)

qam = pynlin.constellations.QAM(32)
qam_symbols = qam.symbols()

g, t = pulse.data()
g = g / g.max()

dt = 1 / baud_rate / samples_per_symbol
t = np.arange(len(t)) * dt

plt.figure()
plt.stem(t * baud_rate, g)


symbols_seq = generate_symbol_sequence(qam_symbols, args.train_length)
train = generate_train(symbols_seq, samples_per_symbol)

pulse_train = scipy.signal.convolve(train, g, method="direct")
recv = pulse_train

t = np.arange(recv.shape[0]) * dt
filter_length = len(g) // 2
shift = 0
truncated_t = t[filter_length + shift : -filter_length]
truncated = recv[filter_length + shift : -filter_length]
sampled_t = truncated_t[::samples_per_symbol]
sampled = truncated[::samples_per_symbol]

plt.figure()
plt.subplot(211)
plt.plot(t * baud_rate, np.real(recv))
plt.plot(t * baud_rate, np.imag(recv))
plt.plot(truncated_t * baud_rate, np.real(truncated))
plt.plot(truncated_t * baud_rate, np.imag(truncated))
plt.plot(
    sampled_t * baud_rate,
    np.real(sampled),
    linestyle="none",
    marker=".",
    color="red",
    markersize=12,
)
plt.plot(
    sampled_t * baud_rate,
    np.imag(sampled),
    linestyle="none",
    marker=".",
    color="red",
    markersize=12,
)

plt.subplot(212)
plt.plot(
    truncated_t,
    np.abs(truncated),
)

print("Sampled: ", len(sampled))

rx_power = np.abs(truncated) ** 2
time_interval = np.max(truncated_t) - np.min(truncated_t)
avg_power = np.mean(rx_power)
plt.figure()
plt.plot(rx_power)
plt.plot(avg_power * np.ones_like(rx_power))
plt.title("Rx power")

print("Average power:", avg_power)

plt.figure()
f, Pxx_den = scipy.signal.periodogram(pulse_train)
plt.semilogy(f, abs(Pxx_den))
# plt.xlabel("frequency [Hz]")
plt.ylabel("PSD [V**2/Hz]")

constellation_avg_power = np.mean(np.abs(qam_symbols) ** 2)
print(f"{constellation_avg_power=}")

plt.figure()
plt.scatter(qam_symbols.real, qam_symbols.imag, label="QAM")
plt.scatter(np.real(sampled), np.imag(sampled), label="Sampled", marker="x")
plt.xlabel(r"$\Re{[a_k]}$")
plt.ylabel(r"$\Im{[a_k]}$")
plt.title("Constellation")
plt.legend()
plt.show()
