import scipy.io
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import plotly.graph_objects as go
from pynlin.utils import oi_law

rc('text', usetex=False)

oi_file = 'oi.mat'
mat = scipy.io.loadmat(oi_file)
oi_full = mat['OI']
wl = mat['wavelenght_array'][0] * 1e-3
# average over the polarizations
oi = np.ndarray((21, 21, 4, 4))
print(wl)

# return starting and ending index of the polarization


def polix(f):
    ll = [2, 4, 4, 2]
    st = [0, 2, 6, 10]
    return (st[f], st[f] + ll[f])


for i in range(4):
    for j in range(4):
        oi[:, :, i, j] = np.average(oi_full[:, :, polix(i)[0]:polix(i)[
                                    1], polix(j)[0]:polix(j)[1]], axis=(2, 3))
np.save('oi.npy', oi)

# quadratic fit of the OI in frequency
oi_fit = -1 * np.ones((6, 4, 4))

x, y = np.meshgrid(wl, wl)
for i in range(4):
    for j in range(4):
        oi_fit[:, i, j] = curve_fit(
            oi_law, (x, y), oi[:, :, 1, 1].ravel(), p0=[0, 1, 0, 1, 0, 1])[0]
np.save('oi_fit.npy', oi_fit)


# =========
# PLOTTING
# =========
plot = True
if plot:
    modes = ["01", "11", "21", "02"]
    n_modes = 4
    # for i in range(n_modes):
    #   plt.plot(wl, oi[0, :, i, :], label=modes)
    #   plt.legend()
    #   plt.title(str(modes[i])+" @ 1400nm")
    #   plt.show()
    #   plt.savefig('media/OI_'+str(i)+'_pump-like.pdf')

    # for i in range(n_modes):
    #   plt.plot(wl, oi[-1, :, i, :], label=modes)
    #   plt.legend()
    #   plt.title(str(modes[i])+" @ 1600nm")
    #   plt.show()
    #   plt.savefig('media/OI_'+str(i)+'sign-like.pdf')
    print("maximum difference over frequency")
    for i in range(n_modes):
        for j in range(i, n_modes):
            oi_slice = oi[:, :, i, j]
            print(str(modes[i]) + '/' + str(modes[j]) + " | %2.2e" %
                  (np.max(oi_slice) - np.min(oi_slice)/np.average(oi_slice)))
            fig = go.Figure(data=go.Contour(z=oi_slice * 1e3,
                                            x=wl,
                                            y=wl,
                                            line_smoothing=0,
                                            contours=dict(
                                                showlabels=True,  # show labels on contours
                                                labelfont=dict(  # label font properties
                                                    size=20,
                                                    color='black',
                                                )
                                            ),
                                            colorbar=dict(
                                                title="*10E-3",
                                                titleside='right'),
                                            ),

                            )
            # apply changes
            fig.update_layout(
                autosize=True,
                width=700,
                height=700,
                font_family="serif",
                yaxis=dict(title=str(modes[i])),
                xaxis=dict(title=str(modes[j])),
                # yaxis_ticksuffix = r"$  ",
                # yaxis_tickprefix = r"$",
                # xaxis_tickprefix = r"$",
                # xaxis_ticksuffix = r"$  ",
                font_size=22
            )
            fig.write_image('media/freq_dep_' +
                            str(modes[i]) + '_' + str(modes[j]) + '.pdf')
            # fig.savefig("test.pdf")

            # oi_slice = oi_law(wl, wl, oi_fit[i, j]...)
            # print(str(modes[i])+'/'+str(modes[j]) +" | %2.2e" % (np.max(oi_slice)-np.min(oi_slice)))
            # fig = go.Figure(data=
            #   go.Contour(z=oi_slice*1e3,
            #             x=wl,
            #             y=wl,
            #             line_smoothing=0,
            #             contours=dict(
            #                     showlabels = True, # show labels on contours
            #                     labelfont = dict( # label font properties
            #                         size = 20,
            #                         color = 'black',
            #                     )
            #             ),
            #             colorbar=dict(
            #                 title="*10E-3",
            #                 titleside='right'),
            #     ),

            #   )
            # 	# apply changes
            # fig.update_layout(
            #     autosize=True,
            #     width= 700,
            #     height=700,
            #     font_family = "serif",
            #     yaxis=dict(title=str(modes[i])),
            #     xaxis=dict(title=str(modes[j])),
            #     # yaxis_ticksuffix = r"$  ",
            #     # yaxis_tickprefix = r"$",
            #     # xaxis_tickprefix = r"$",
            #     # xaxis_ticksuffix = r"$  ",
            #     font_size = 22
            # )
            # fig.write_image('media/freq_dep_'+str(modes[i])+'_'+str(modes[j])+'.pdf')
            # # fig.savefig("test.pdf")
