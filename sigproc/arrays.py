#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sigproc
from obspy.signal.array_analysis import array_transff_freqslowness, array_transff_wavenumber
import math
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize


def correlation_analysis(st, flow, fhigh, plotfit=True, plotdots=True):
    """
    Check correlation between stations in an array for a given
    frequency band or range of frequency bands
    INPUTS
    st - obspy stream with coordinates embedded in
    tr.stats.coordinates = AttribDict({'x': 10., 'y': 10., 'elevation': 0.})
    where x and y are in km, or x and y
    NOT IMPLEMENTED YET x and y can be replaced by lat and lon in
    decimal degrees
    flow - float or numpy array of lower range of frequency bands to check
    fhigh - float or numpy array of floats of upper range of frequency
    bands to check
    """
    # ADD CHECK FOR COORDINATES

    rowlen = len(st)*(len(st)-1)/2
    collen = len(flow)
    distkm = np.zeros((rowlen, collen))
    corr = distkm.copy()
    lagsec = distkm.copy()
    col = 0
    labels = []
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    for fmin, fmax in zip(flow, fhigh):
        stfilt = st.copy()
        stfilt.detrend('linear')
        stfilt.taper(max_percentage=0.05, type='cosine')
        stfilt.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True)
        labels.append('%1.1f - %1.1f Hz' % (fmin, fmax))
        row = 0
        for j in np.arange(0, len(stfilt)):
            for k in np.arange(j+1, len(stfilt)):
                distkm[row, col] = (np.sqrt((stfilt[j].stats.coordinates.x-stfilt[k].stats.coordinates.x)**2 + (stfilt[j].stats.coordinates.y-stfilt[k].stats.coordinates.y)**2 + (stfilt[j].stats.coordinates.elevation-stfilt[k].stats.coordinates.elevation)**2))
                maxcor, maxlag, dt = sigproc.xcorrnorm(stfilt[j], stfilt[k])
                corr[row, col] = maxcor
                lagsec[row, col] = np.abs(dt)
                row += 1
        if plotdots:
            ax.plot(distkm[:, col], corr[:, col], '.', label=labels[col])
            ax1.plot(distkm[:, col], lagsec[:, col], '.', label=labels[col])
        if plotfit:
            p = np.polyfit(distkm[:, col], corr[:, col], 1)
            z = np.poly1d(p)
            dists = np.linspace(0., distkm[:, col].max(), 10)
            ax.plot(dists, z(dists), label=labels[col])
            p1 = np.polyfit(distkm[:, col], lagsec[:, col], 1)
            z1 = np.poly1d(p1)
            ax1.plot(dists, z1(dists), label=labels[col])
        col += 1

    ax.legend(fontsize=8)
    ax.set_ylabel('Xcorr')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Lag (sec)')

    plt.show()

    return distkm, corr, lagsec


def extract_coords(st):
    """
    Extract coordinates for Array response function plotting
    """
    coords = []
    for trace in st:
        coords.append(np.array([trace.stats.coordinates.latitude, trace.stats.coordinates.longitude, trace.stats.coordinates.elevation]))
    coords = np.array(coords)
    return coords


def plotarray(coords, coordsys='xy'):
    """
    ADD LAT/LON CAPABILITY SOMETIME
    """
    plt.plot([coord[0] for coord in coords], [coord[1] for coord in coords], 'o')
    plt.axis('equal')
    plt.show()
    return coord


def plotARF_slowaz(coords, slim, sstep, freqlow, freqhigh, fstep, coordsys='xy'):
    """
    Add ability to plot range of frequency ranges
    """
    transff = array_transff_freqslowness(coords, slim, sstep, freqlow, freqhigh, fstep, coordsys=coordsys)
    xgrid = np.arange(-slim, slim+sstep, sstep),
    slow = np.empty((len(xgrid), len(xgrid))),
    baz = slow.copy(),
    for i in np.arange(len(xgrid)):
        for j in np.arange(len(xgrid)):
            # compute baz, slow,
            slow_x = xgrid[i],
            slow_y = xgrid[j],
            slow[i, j] = np.sqrt(slow_x ** 2 + slow_y ** 2),
            if slow[i, j] < 1e-8:
                slow[i, j] = 1e-8
            azimut = 180 * math.atan2(slow_x, slow_y) / math.pi,
            baz[i, j] = azimut % -360 + 180,
    baz[baz < 0.0] += 360,
    # transform to radian,
    baz = np.radians(baz),
    cmap = cm.RdYlBu,
    fig = plt.figure(figsize=(8, 8)),
    cax = fig.add_axes([0.85, 0.2, 0.05, 0.5]),
    ax = fig.add_axes([0.10, 0.1, 0.70, 0.7], polar=True),
    ax.pcolormesh(baz, slow, transff, vmin=0., vmax=1., cmap=cmap),
    ax.set_theta_direction(-1),
    ax.set_theta_zero_location("N")
    ax.set_ylim(0, slim),
    ax.grid(),
    ColorbarBase(cax, cmap=cmap, norm=Normalize(vmin=0., vmax=1.))
    plt.show()


def plotARF_k(coords, klim, kstep, coordsys='xy'):
    transff = array_transff_wavenumber(coords, klim, kstep, coordsys=coordsys),
    xgrid = np.arange(-klim, klim+kstep, kstep),
    cmap = cm.RdYlBu,
    fig = plt.figure(figsize=(8, 8)),
    cax = fig.add_axes([0.85, 0.2, 0.05, 0.5]),
    ax = fig.add_axes([0.10, 0.1, 0.70, 0.7]),
    ax.pcolormesh(xgrid, xgrid, transff, vmin=0., vmax=1.0, cmap=cmap),
    ax.grid(),
    ax.set_xlabel('kx (1/km)'),
    ax.set_ylabel('ky (1/km)'),
    ColorbarBase(cax, cmap=cmap, norm=Normalize(vmin=0., vmax=1.0)),
    plt.show()


def beamform_slowaz(st, slim, sstep, freqlow, freqhigh, fstep, win_len, stime=None, etime=None, win_frac=0.05, coordsys='xy', moviefolder=None, movie=True):
    import glob
    import os
    from sigproc import sigproc

    if moviefolder is None:
        moviefolder = os.get_cwd()
    if stime is None:
        stime = st[0].stats.starttime
    if etime is None:
        etime = st[0].stats.starttime
    if movie:
        store = dump

    def dump(pow_map, apow_map, i):
        """Example function to use with `store` kwarg in
        :func:`~obspy.signal.array_analysis.array_processing`.
        """
        np.savez(moviefolder+'/pow_map_%d.npz' % i, pow_map)

    kwargs = dict(
        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll_x=-slim, slm_x=slim, sll_y=-slim, slm_y=slim, sl_s=sstep,
        # sliding window properties
        win_len=win_len, win_frac=win_frac,
        # frequency properties
        frqlow=freqlow, frqhigh=freqhigh, prewhiten=0,
        # restrict output
        semb_thres=-1e9, vel_thres=-1e9,
        stime=stime,
        etime=etime, coordsys=coordsys, store=None)

    out = array_processing(st, **kwargs)

    # Plot 1
    labels = ['rel.power', 'abs.power', 'baz', 'slow']

    xlocator = mdates.AutoDateLocator()
    fig = plt.figure()
    for i, lab in enumerate(labels):
        ax = fig.add_subplot(4, 1, i + 1)
        ax.scatter(out[:, 0], out[:, i + 1], c=out[:, 1], alpha=0.6,
                   edgecolors='none')
        ax.set_ylabel(lab)
        ax.set_xlim(out[0, 0], out[-1, 0])
        ax.set_ylim(out[:, i + 1].min(), out[:, i + 1].max())
        ax.xaxis.set_major_locator(xlocator)
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))

    #fig.suptitle('')
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.2, hspace=0)
    plt.show()

    # Plot 2

    cmap = cm.hot_r

    # make output human readable, adjust backazimuth to values between 0 and 360
    t, rel_power, abs_power, baz, slow = out.T
    baz[baz < 0.0] += 360

    # choose number of fractions in plot (desirably 360 degree/N is an integer!)
    N = 36
    N2 = 30
    abins = np.arange(N + 1) * 360. / N
    sbins = np.linspace(0, 4, N2 + 1)

    # sum rel power in bins given by abins and sbins
    hist, baz_edges, sl_edges = np.histogram2d(baz, slow, bins=[abins, sbins], weights=rel_power)

    # transform to radian
    baz_edges = np.radians(baz_edges)

    # add polar and colorbar axes
    fig = plt.figure(figsize=(8, 8))
    cax = fig.add_axes([0.85, 0.2, 0.05, 0.5])
    ax = fig.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')

    dh = abs(sl_edges[1] - sl_edges[0])
    dw = abs(baz_edges[1] - baz_edges[0])

    # circle through backazimuth
    for i, row in enumerate(hist):
        bars = ax.bar(left=(i * dw) * np.ones(N2),
                      height=dh * np.ones(N2),
                      width=dw, bottom=dh * np.arange(N2), color=cmap(row / hist.max()))

    ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax.set_xticklabels(['N', 'E', 'S', 'W'])

    # set slowness limits
    ax.set_ylim(0, slm_x)
    [i.set_color('grey') for i in ax.get_yticklabels()]
    ColorbarBase(cax, cmap=cmap, norm=Normalize(vmin=hist.min(), vmax=hist.max()))
    plt.show()

    if movie:
        xgrid = np.arange(sll_x, slm_x+sl_s, sl_s)
        ygrid = np.arange(sll_y, slm_y+sl_s, sl_s)
        slow = np.empty((len(xgrid), len(ygrid)))
        baz = slow.copy()
        for i in np.arange(len(xgrid)):
            for j in np.arange(len(ygrid)):
                # compute baz, slow
                slow_x = xgrid[i]
                slow_y = ygrid[j]
                slow[i, j] = np.sqrt(slow_x ** 2 + slow_y ** 2)
                if slow[i, j] < 1e-8:
                    slow[i, j] = 1e-8
                azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
                baz[i, j] = azimut % -360 + 180
        baz[baz < 0.0] += 360
        # transform to radian
        baz = np.radians(baz)

        x, y = np.meshgrid(xgrid, ygrid)

        pow_map_mean = np.array((x, y))
        findind = 0
        
        stfilt = traces.copy()
        stfilt.filter('bandpass', freqmin=frqlow, freqmax=frqhigh)
        
        t, rel_power, abs_power, baz1, slowness = out.T
        baz1[baz1 < 0.0] += 360
        for i, t1 in enumerate(t):
        filen = glob.glob(moviefolder+'/pow_map_%i.npz' % findind)
        plt.close('all')
        f = np.load(filen[0])
        pow_map = f['arr_0']
        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_axes([0.05, 0.05, 0.25, 0.9], polar=True)
        ax.pcolormesh(baz, slow, pow_map, vmin=0.5, vmax=1., cmap=cmap)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_ylim(0, slm_x)
        
        ix, iy = np.unravel_index(pow_map.argmax(), pow_map.shape)
        az = 180 * math.atan2(xgrid[ix], ygrid[iy]) / math.pi
        bazimut = az % -360 + 180
        if bazimut < 0.0:
            bazimut += 360
        slow1 = np.sqrt(xgrid[ix]**2 + ygrid[iy]**2)
        ax.plot(np.radians(bazimut), slow1, 'xk')
        ax.text(np.radians(bazimut), slow1, '  %1.1f km/s\\n %1.0f deg' % (1./slow1, bazimut))
        ax.grid()
        
        cax = fig.add_axes([0.32, 0.15, 0.01, 0.7])
        ColorbarBase(cax, cmap=cmap, norm=Normalize(vmin=0.5, vmax=1.))
        ax1 = fig.add_axes([0.37, 0.05, 0.58, 0.9])
        tvec = sigproc.maketvec(stfilt[0])
        ax1.plot(tvec, stfilt[0].data/max(stfilt[0].data), 'k', label=stfilt[0].stats.station)
        ax1.plot(tvec, stfilt[1].data/max(stfilt[1].data) + 1.5, 'k', label=stfilt[1].stats.station)
        ax1.plot(tvec, stfilt[2].data/max(stfilt[2].data) + 3., 'k', label=stfilt[2].stats.station)
        sec = UTCDateTime(mdates.num2date(t[i]))-stfilt[0].stats.starttime
        ax1.vlines(sec, -1, 4, color='r')
        ax1.vlines(sec + win_len, -1, 4, color='r')
        plt.title('Max at %3.0f degrees, speed of %1.1f km/s - tstart %1.0f' % (baz1[i], 1/slowness[i], sec))
        ax1.set_ylim(-1, 4)
        plt.savefig(('/Users/kallstadt/Flume/beamformtest/img%03d.png') % (i,))
        findind += int(win_len*st[0].stats.sampling_rate*win_frac)
        plt.show()
        if 1:
        #turn into mpg
        os.chdir('/Users/kallstadt/Flume/beamformtest/')
        #os.system('ffmpeg -f image2 -start_number 1 -r 3 -i img%03d.png -y -c:v libx264 -vf \format=yuv420p\ Oso_animation.mp4')
        os.system('ffmpeg -f image2 -start_number 0 -r 4 -i img%03d.png -y -c:v libx264 -vf \format=yuv420p\ Beamform_test.mp4')
        #os.system('ffmpeg -i Oso_animation.mp4 -qscale 0 Oso_animation.wmv')
        os.chdir('..')
