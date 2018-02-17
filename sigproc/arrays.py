#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sigproc
from obspy.signal.array_analysis import array_transff_freqslowness, array_transff_wavenumber, array_processing, get_geometry, get_timeshift, get_spoint, get_geometry
from obspy.signal.util import util_geo_km, next_pow_2
from obspy import UTCDateTime, Stream
from obspy.signal.headers import clibsignal
import math
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.dates as mdates
import glob
import os
from scipy.fftpack import fft, ifft, ifftshift
from reviewData.reviewData import unique_list


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
    if type(flow) is float:
        collen = 1
        flow = [flow]  # turn into lists so later iteration doesn't crash
        fhigh = [fhigh]
    else:
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
                maxcor, maxlag, maxdt, cc, lags, tlags = sigproc.xcorrnorm(stfilt[j], stfilt[k])
                corr[row, col] = maxcor
                lagsec[row, col] = np.abs(maxdt)
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


def extract_coords(st, inunits='m', outunits='km'):
    """
    Extract coordinates for Array response function plotting and convert units if needed
    """
    coords = []
    trace = st[0]
    if 'latitude' in trace.stats.coordinates:
        for trace in st:
            coords.append(np.array([trace.stats.coordinates.longitude, trace.stats.coordinates.latitude, trace.stats.coordinates.elevation]))
        coordsys = 'lonlat'
        coords = np.array(coords)
    elif 'x' in trace.stats.coordinates:
        for trace in st:
            coords.append(np.array([trace.stats.coordinates.x, trace.stats.coordinates.y, trace.stats.coordinates.elevation]))
        coordsys = 'xy'
        if inunits == 'm' and outunits == 'km':
            coords = np.array(coords)/1000.
        elif inunits == 'km' and outunits == 'm':
            coords = np.array(coords)*1000.
        else:
            coords = np.array(coords)
    else:
        print('no coordinates found in stream')
        return

    return coords, coordsys


def plotarray(stcoord, inunits='m', plotunits='km', sourcecoords=None, stalabels=None):
    """
    stcoord = stream or coords extracted in format from extract_coords
    sourcecoords = lat,lon or x,y (should be in same units as stcoord)
    inunits 'm' 'km' or 'deg'
    """

    coords = []
    tempcoords = []

    if type(stcoord) is Stream:
        if stalabels is None:
            stalabels = []
        for trace in stcoord:
            if inunits == 'deg':
                tempcoords.append(np.array([trace.stats.coordinates.longitude, trace.stats.coordinates.latitude, trace.stats.coordinates.elevation]))
                stalabels.append(trace.stats.station)
            else:
                tempcoords.append(np.array([trace.stats.coordinates.x, trace.stats.coordinates.y, trace.stats.coordinates.elevation]))
                stalabels.append(trace.stats.station)
    else:
        tempcoords = stcoord
    tempcoords = np.array(tempcoords)

    if inunits == 'deg':
        lons = np.array([coord[0] for coord in tempcoords])
        lats = np.array([coord[1] for coord in tempcoords])
        for coord in tempcoords:
            x, y = util_geo_km(lons.min(), lats.min(), coord[0], coord[1])
            coords.append(np.array([x, y, coord[2]]))
        if plotunits == 'm':
            coords = np.array(coords)*1000.
        else:
            coords = np.array(coords)
        if sourcecoords is not None:
            sx, sy = util_geo_km(lons.min(), lats.min(), sourcecoords[0], sourcecoords[1])

    elif inunits == 'm' and plotunits == 'km':
        coords = np.array(tempcoords)/1000.
        if sourcecoords is not None:
            sx, sy = sourcecoords/1000.
    elif inunits == 'km' and plotunits == 'm':
        coords = np.array(tempcoords)*1000.
        if sourcecoords is not None:
            sx, sy = sourcecoords*1000.
    else:
        coords = np.array(tempcoords)
        if sourcecoords is not None:
            sx, sy = sourcecoords

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([coord[0] for coord in coords], [coord[1] for coord in coords], 'o')
    if stalabels is not None:
        for coord, sta in zip(coords, stalabels):
            ax.text(coord[0], coord[1], sta, fontsize=8)
    if sourcecoords is not None:
        ax.plot(sx, sy, '*k', markersize=20)
        xcenter = np.array([coord[0] for coord in coords]).mean()
        ycenter = np.array([coord[1] for coord in coords]).mean()
        az = 180 * math.atan2((xcenter-sx), (ycenter-sy)) / math.pi
        baz = az % -360 + 180
        if baz < 0.0:
            baz += 360
        ax.set_title('Backazimuth %0.0f degrees' % baz)
    else:
        sx = None
        sy = None
        baz = None
        az = None
    plt.axis('equal')
    ax.set_xlabel('x distance (%s)' % plotunits)
    ax.set_ylabel('y distance (%s)' % plotunits)
    plt.show()
    # ADD AZIMUTH TO CENTER OF ARRAY
    return coords, sx, sy, baz, az


def plotARF_slowaz(coords, slim, sstep, freqlow, freqhigh, fstep, coordsys='xy'):
    """
    Add ability to plot range of frequency ranges
    """
    transff = array_transff_freqslowness(coords, slim, sstep, freqlow, freqhigh, fstep, coordsys=coordsys)
    xgrid = np.arange(-slim, slim+sstep, sstep)
    slow = np.empty((len(xgrid), len(xgrid)))
    baz = slow.copy()
    for i in np.arange(len(xgrid)):
        for j in np.arange(len(xgrid)):
            # compute baz, slow
            slow_x = xgrid[i]
            slow_y = xgrid[j]
            slow[i, j] = np.sqrt(slow_x ** 2 + slow_y ** 2)
            if slow[i, j] < 1e-8:
                slow[i, j] = 1e-8
            azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
            baz[i, j] = azimut % -360 + 180
    baz[baz < 0.0] += 360
    # transform to radian,
    baz = np.radians(baz)
    cmap = cm.RdYlBu
    fig = plt.figure(figsize=(8, 8))
    cax = fig.add_axes([0.85, 0.2, 0.05, 0.5])
    ax = fig.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)
    ax.pcolormesh(baz, slow, transff, vmin=0., vmax=1., cmap=cmap)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.set_ylim(0, slim)
    ax.grid()
    fig.suptitle('Array response function')
    ColorbarBase(cax, cmap=cmap, norm=Normalize(vmin=0., vmax=1.))
    plt.show()


def plotARF_k(coords, klim, kstep, coordsys='xy'):
    transff = array_transff_wavenumber(coords, klim, kstep, coordsys=coordsys)
    xgrid = np.arange(-klim, klim+kstep, kstep)
    cmap = cm.RdYlBu
    fig = plt.figure(figsize=(8, 8))
    cax = fig.add_axes([0.85, 0.2, 0.05, 0.5])
    ax = fig.add_axes([0.10, 0.1, 0.70, 0.7])
    ax.pcolormesh(xgrid, xgrid, transff, vmin=0., vmax=1.0, cmap=cmap)
    ax.grid()
    ax.set_xlabel('kx (1/km)')
    ax.set_ylabel('ky (1/km)')
    fig.suptitle('Array response function')
    ColorbarBase(cax, cmap=cmap, norm=Normalize(vmin=0., vmax=1.0))
    plt.show()


def beamform_plane(st, sll_x, slm_x, sll_y, slm_y, sstep, freqlow, freqhigh,
                   win_len, stime=None, etime=None, win_frac=0.05, coordsys='xy',
                   outfolder=None, movie=True, savemovieimg=False, plottype='slowaz',
                   showplots=True, saveplots=False, plotlabel='', saveall=False,
                   timestamp='centered'):
    """
    MAKE CHOICE TO USE Sx Sy or S A in PLOTTING
    plotype = 'slowaz' or 'wavenum'
    NEED ffmpeg for movie making to work, otherwise will just get the images
    timestamp (str): 'centered', the time stamp will be at the middle of the
        time window, 'left', the time stampe will be at the beginning of the 
        time window
    """
    if outfolder is None:
        outfolder = os.getcwd()

    def dump(pow_map, apow_map, i):
        """Example function to use with `store` kwarg in
        :func:`~obspy.signal.array_analysis.array_processing`.
        """
        np.savez(os.path.join(outfolder, 'pow_map_%d.npz' % i), pow_map)
        np.savez(os.path.join(outfolder, 'apow_map_%d.npz' % i), apow_map)

    rates = [tr.stats.sampling_rate for tr in st]
    sampling_rate = unique_list(rates)
    if len(sampling_rate) > 1:
        raise Exception('all traces in st must have the same sampling rate')
    else:
        sampling_rate = sampling_rate[0]
    lens = [len(tr) for tr in st]
    lens = unique_list(lens)
    if len(lens) > 1:
        raise Exception('all traces in st must have the same number of samples')

    if stime is None:
        stime = np.max([tr.stats.starttime for tr in st])
    if etime is None:
        etime = np.min([tr.stats.endtime for tr in st])
    
    if movie or saveall:
        store = dump
    else:
        store = None

    kwargs = dict(
        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll_x=sll_x, slm_x=slm_x, sll_y=sll_y, slm_y=slm_y, sl_s=sstep,
        # sliding window properties
        win_len=win_len, win_frac=win_frac,
        # frequency properties
        frqlow=freqlow, frqhigh=freqhigh, prewhiten=0,
        # restrict output
        semb_thres=-1e9, vel_thres=-1e9,
        stime=stime,
        etime=etime, coordsys=coordsys, store=store)

    out = array_processing(st, **kwargs)

    # make output human readable, adjust backazimuth to values between 0 and 360
    t, rel_power, abs_power, baz, slow = out.T
    baz[baz < 0.0] += 360

    if saveplots or showplots:
        # Plot 1
        labels = ['rel.power', 'abs.power', 'baz', 'slow']
    
        xlocator = mdates.AutoDateLocator()
        fig1 = plt.figure()
        for i, lab in enumerate(labels):
            ax = fig1.add_subplot(4, 1, i + 1)
            ax.scatter(out[:, 0], out[:, i + 1], c=out[:, 1], alpha=0.6,
                       edgecolors='none')
            ax.set_ylabel(lab)
            ax.set_xlim(out[0, 0], out[-1, 0])
            ax.set_ylim(out[:, i + 1].min(), out[:, i + 1].max())
            ax.xaxis.set_major_locator(xlocator)
            ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))
    
        fig1.autofmt_xdate()
        fig1.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.2, hspace=0)
    
        # Plot 2
    
        cmap = cm.hot_r
    
        # choose number of fractions in plot (desirably 360 degree/N is an integer!)
        N = 36
        N2 = 30
        abins = np.arange(N + 1) * 360. / N
        sbins = np.linspace(0, np.sqrt(slm_x**2 + slm_y**2), N2 + 1)
    
        # sum rel power in bins given by abins and sbins
        hist, baz_edges, sl_edges = np.histogram2d(baz, slow, bins=[abins, sbins], weights=rel_power)
    
        # transform to radian
        baz_edges = np.radians(baz_edges)
    
        # add polar and colorbar axes
        fig2 = plt.figure(figsize=(8, 8))
        cax = fig2.add_axes([0.85, 0.2, 0.05, 0.5])
        ax = fig2.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
    
        dh = abs(sl_edges[1] - sl_edges[0])
        dw = abs(baz_edges[1] - baz_edges[0])
    
        # circle through backazimuth
        for i, row in enumerate(hist):
            bars = ax.bar(left=(i * dw) * np.ones(N2),
                          height=dh * np.ones(N2),
                          width=dw, bottom=dh * np.arange(N2), color=cmap(row / hist.max()))
    
        ax.set_xticks(np.linspace(0, 2 * np.pi, np.sqrt(slm_x**2 + slm_y**2), endpoint=False))
        ax.set_xticklabels(['N', 'E', 'S', 'W'])
    
        # set slowness limits
        ax.set_ylim(0, np.sqrt(slm_x**2 + slm_y**2))
        [i.set_color('grey') for i in ax.get_yticklabels()]
        ColorbarBase(cax, cmap=cmap, norm=Normalize(vmin=hist.min(), vmax=hist.max()))
    
        if showplots:
            plt.show()
        if saveplots:
            fig1.savefig('%s/%s-%s.png' % (outfolder, 'timeplot', plotlabel))
            fig2.savefig('%s/%s-%s.png' % (outfolder, 'overallplot', plotlabel))

    if movie:
        cmap = cm.RdYlBu
        xgrid = np.arange(sll_x, slm_x+sstep, sstep)
        ygrid = np.arange(sll_y, slm_y+sstep, sstep)
        slow2 = np.empty((len(xgrid), len(ygrid)))
        baz2 = slow2.copy()
        for i in np.arange(len(xgrid)):
            for j in np.arange(len(ygrid)):
                # compute baz, slow
                slow_x = xgrid[i]
                slow_y = ygrid[j]
                slow2[i, j] = np.sqrt(slow_x ** 2 + slow_y ** 2)
                if slow2[i, j] < 1e-8:
                    slow2[i, j] = 1e-8
                azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
                baz2[i, j] = azimut % -360 + 180
        baz2[baz2 < 0.0] += 360
        # transform to radian
        baz2 = np.radians(baz2)

        x, y = np.meshgrid(xgrid, ygrid)

        #pow_map_mean = np.array((x, y))
        findind = 0

        stfilt = st.copy()
        stfilt.filter('bandpass', freqmin=freqlow, freqmax=freqhigh)
        stfilt.trim(stime-5., etime+5., pad=True, fill_value=0.)

        for i, t1 in enumerate(t):
            filen = glob.glob(outfolder+'/pow_map_%i.npz' % findind)
            f = np.load(filen[0])
            pow_map = f['arr_0']
            fig = plt.figure(figsize=(18, 6))
            ax = fig.add_axes([0.05, 0.05, 0.25, 0.9], polar=True)
            ax.pcolormesh(baz2, slow2, pow_map, vmin=0., vmax=1., cmap=cmap)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location('N')
            #ax.set_ylim(0, np.sqrt(slm_x**2 + slm_y**2))
            ax.set_ylim(0, np.max([slm_x, slm_y]))

            ix, iy = np.unravel_index(pow_map.argmax(), pow_map.shape)
            az = 180 * math.atan2(xgrid[ix], ygrid[iy]) / math.pi
            bazimut = az % -360 + 180
            if bazimut < 0.0:
                bazimut += 360
            slow1 = np.sqrt(xgrid[ix]**2 + ygrid[iy]**2)
            ax.plot(np.radians(bazimut), slow1, 'xk')
            ax.text(np.radians(bazimut), slow1, '  %1.1f km/s\\n %1.0f deg' % (1./slow1, bazimut))
            ax.grid()
            plt.draw()

            cax = fig.add_axes([0.32, 0.15, 0.01, 0.7])
            ColorbarBase(cax, cmap=cmap, norm=Normalize(vmin=0.0, vmax=1.))
            ax1 = fig.add_axes([0.37, 0.05, 0.58, 0.9])
            tvec = sigproc.maketvec(stfilt[0])
            #import pdb; pdb.set_trace()
            ax1.plot(tvec, stfilt[0].data/max(stfilt[0].data), 'k', label=stfilt[0].stats.station)
            ax1.plot(tvec, stfilt[1].data/max(stfilt[1].data) + 1.5, 'k', label=stfilt[1].stats.station)
            ax1.plot(tvec, stfilt[2].data/max(stfilt[2].data) + 3., 'k', label=stfilt[2].stats.station)
            sec = UTCDateTime(mdates.num2date(t1))-stfilt[0].stats.starttime
            ax1.vlines(sec, -1, 4, color='r')
            ax1.vlines(sec + win_len, -1, 4, color='r')
            plt.title('Max at %3.0f degrees, speed of %1.1f km/s - tstart %1.0f' % (baz[i], 1/slow[i], sec))
            ax1.set_ylim(-1, 4)
            plt.savefig(('%s/img%03d.png') % (outfolder, i))
            findind += int(win_len*st[0].stats.sampling_rate*win_frac)
            #plt.draw()
            plt.show()
            plt.close(fig)

        #turn into mpg
        origdir = os.getcwd()
        os.chdir(outfolder)
        os.system('ffmpeg -f image2 -start_number 0 -r 4 -i img%03d.png -y -c:v libx264 -vf "format=yuv420p" beammovie.mp4')
        # Clean up
        delfiles = glob.glob(os.path.join(outfolder, 'pow_map_*.npz'))
        if not saveall:
            for df in delfiles:
                os.remove(df)
            if savemovieimg is False:
                delfiles = glob.glob(os.path.join(outfolder, 'img*png'))
                for df in delfiles:
                    os.remove(df)
        os.chdir(origdir)
    
    # shift timestamp if necessary (obspy puts it at the beginning of the window)
    if timestamp == 'centered':
        numsecindays = win_len/(24.*60*60)
        t = t + numsecindays
    
    return t, rel_power, abs_power, baz, slow


def backproject(st, v, tshifts, freqmin, freqmax, winlen, overlap, gridx, gridy, gridz=None, starttime=None, endtime=None, relshift=0., coords=None, ds=False):
    ## COULD EASILY OUTPUT ALL FREQUENCIES, BUT ONLY VALID FOR THE ONE THE TIMESHIFTS OR SPEED ARE FOR
    """
    st should have coordinates embedded, if not, coords need to be supplied in teh form of ?
    v, constant velocity to use to estimate travel times in half space set to None if tshifts are specified
    tshifts, time shifts, in seconds, from each station to each grid point in same order as traces in st, set to None if using constant velocity
    freqmin - lowest frequency to consider
    freqmax - higher frequency to consider
    gridxyz - grid of points to search over for backprojection
    winlen - window length in seconds
    overlap - amount to overlap windows from 0 (no overlap) to 1 (completely overlapped)
    coords - station coordinates, if not embedded in stream, only need for mapping and for computing travel times, not required if tshifts are provided. Should be in form of dictionary {sta: (x,y,elev)}, be sure units are consistent between v and coords
    ds - If true, will downsample to 1/(fhigh*2)
    starttime - start time in UTC of analysis, if None, will use start time of st
    endtime - same as above but end time
    relshift - relative shift to apply to all stations to account for travel time to first station
    """
    # Initial data processing
    samprates = [trace.stats.sampling_rate for trace in st]
    if np.mean(samprates) != samprates[0] and ds is False:
        print('sample rates are not all equal, resampling to minimum sample rate')
        st = st.resample(np.min(samprates))
    if ds is True:
        st = st.resample(1./(freqmax*2.))
    dt = st[0].stats.sampling_rate
    if starttime is None:
        starttime = np.min([trace.stats.starttime for trace in st])
    if endtime is None:
        endtime = np.max([trace.stats.endtime for trace in st])
    st.trim(starttime, endtime, pad=True)  # turns into masked array if needs to pad, might need to check if this breaks things

    nsta = len(st)

    if gridz is None:
        gridz = np.zeros(np.shape(gridx))  # if no elevations provided, assume all at zero elevation
    # Pull out coords for later usage
    sx = []
    sy = []
    selev = []
    names = []
    chan = []
    if coords is None:
        for trace in st:
            sx.append(trace.stats.coordinates['x'])
            sy.append(trace.stats.coordinates['y'])
            selev.append(trace.stats.coordinates['elevation'])
            names.append(trace.stats.station)
            chan.append(trace.stats.channel)
    else:
        for trace in st:
            sx.append(coords[trace.stats.station][0])
            sy.append(coords[trace.stats.station][1])
            selev.append(coords[trace.stats.station][2])
            names.append(trace.stats.station)
            chan.append(trace.stats.channel)

    nwins = int((endtime-starttime)/(winlen*(1.-overlap))) - 3  # look into why 3
    incr = (1.-overlap)*winlen
    sttm = np.arange(0., incr*nwins, incr)

    winlensamp = int(np.round(winlen*dt))
    nfft = 2*next_pow_2(winlen*dt)
    freqs = np.fft.fftfreq(nfft, 1/dt)
    freqsubl = len(freqs[(freqs >= freqmin) & (freqs <= freqmax)])

    power = np.zeros((len(gridx), nwins, freqsubl))
    powernon = power.copy()
    meanpow = np.zeros((len(gridx), nwins))
    # Compute stack power at each hammer location for each point in time (need to figure out how to do ARF for this)
    for n in np.arange(nwins):
        reftime = starttime + sttm[n]
        for i, (gx, gy, gz) in enumerate(zip(gridx, gridy, gridz)):
            # Figure out what shifts are
            if v is not None and tshifts is None:
                shifts = np.sqrt((sx-gx)**2 + (sy-gy)**2 + (selev-gz)**2)/v
            elif tshifts is not None and v is None:
                shifts = tshifts + relshift
            else:
                print('neither tshifts or v defined properly')
            # Cut out window at relative start time
            sttemp = st.copy().trim(starttime=reftime)
            # Apply shifts, in time domain for grid point i
            extract = np.zeros((len(sttemp), freqsubl)) + 1j*np.zeros((len(sttemp), freqsubl))
            extractnon = extract.copy()
            k = 0
            for tr, shift in zip(sttemp, shifts):
                tr.detrend('demean')
                tr.taper(max_percentage=0.05, type='cosine')
                x = tr.data[:winlensamp]
                s = int(np.round(shift * tr.stats.sampling_rate))
                N = nfft
                r = np.floor(N/2)+1
                f = (np.arange(1, N+1)-r)/(N/2)
                p = np.exp(-1j*s*np.pi*f)  # seems like it needs negative sign to shift in the right direction...
                y = fft(x, nfft)*ifftshift(p)
                ynon = fft(x, nfft)
                # whitening?
                #y = y/np.abs(y)
                #indx = np.where((freqs >= freqmin) & (freqs <= freqmax))
                extract[k, :] = y[(freqs >= freqmin) & (freqs <= freqmax)]
                extractnon[k, :] = ynon[(freqs >= freqmin) & (freqs <= freqmax)]
                #import pdb;pdb.set_trace()
                k += 1
                # extract mean beam power over this frequency range and add to cumulative total for this point
            power[i, n, :] = ((1./nsta) * np.abs(np.sum(extract, 0)))**2
            powernon[i, n, :] = ((1./nsta) * np.abs(np.sum(extractnon, 0)))**2
            meanpow[i, n] = np.mean(power[i, n, :])/np.mean(powernon[i, n, :])
        print('%i of %i' % (n, nwins))

    fvec = freqs[(freqs >= freqmin) & (freqs <= freqmax)]
    tvec = sttm + incr/2.
    return power, meanpow, tvec, fvec


def Almendros_beam(st, win_len, Smin, Smax, Sstep, Amin, Amax, Astep, spherical=False,
                   Rmin=None, Rmax=None, Rstep=None, stime=None, etime=None, overlap=0.8,
                   outfolder=None, subsample=True):
    """Plane and spherical wavefront beamforming using approch of Almendros et al 1999
    
    uses cross correlation in time domain instead of frequency domain as
    fk beamforming does, this allows for smaller time windows to be used

    st: obspy stream of array data with station coordinates attached in
        cartesian coords, km units (not latlon)
    win_len: window length in seconds
    Smin: minimum slowness (s/km)
    Smax: maximum slowness (s/km)
    Sstep: slowness step (s/km)
    Amin: minimum azimuth (degrees)
    Amax: maximum azimuth (degrees)
    Astep: azimuth step (degrees)
    Dmin: 

        
    subsample (bool): if True, will shift timing with subsample accuracy
        by using linear interpolation (may slow things down)
    """
    if outfolder is None:
        outfolder = os.getcwd()
    starttime, endtime, sampling_rate, numsamp = checksync(st)
    numsta = len(st)
    slows = np.linspace(Smin, Smax, Sstep)
    baz = np.linspace(Amin, Amax, Astep)

    geometry = get_geometry(st, coordsys='xy')  # returns matrix of x,y,z, each row corresponds to each station in st, relative to center of array (0,0,0)
    tsb = timeshifts(geometry, baz, slows)

    alldata = np.vstack([trace.data for trace in st])

    beams = np.zeros(len(baz), (len(slows), numsamp))

    # iterate over slownesses
    for j in range(len(baz)):
        for k in range(len(slows)):
            temp = np.zeros(np.shape(alldata))
            # get shifts for this slowness in number of samples
            shifts = tsb[:, j, k] * sampling_rate
            if subsample: # interpolate in shift for subsample accuracy
                for i in range(numsta):
                    tv = np.arange(0., numsamp)
                    temp[i, :] = np.interp(tv - shifts[i], tv, alldata[i, :])
            else: # otherwise round to nearest sample
                shifts = np.round(shifts)
                for i in range(numsta):
                    temp[i, :] = np.roll(alldata[i, :], int(shifts[i]))
            # Compute zero-lag cross correlation over entire matrix
            STARTHERE
            fullmat = 1./numsta**2 * alldata*temp/(np.sqrt())
            
            


    for i, t1 in enumerate(t):
        pass
        # Check if rel_power was high enough

        # If it was, define search area for spherical approximation

        # Save this as a polygon for plotting
        
        # Loop over all distances - CAN I DO THIS WITH MATRIX OPERATIONS SOMEHOW?

            # Loop over all S

                # Loop over all A

                    # Apply shifts to matrix of data, then cut out slice corresponding to t1

                    # Compute array-averaged zero-lag normalized correlations

                    # Save in correct part of result matrix


        # Plot space and time overview (map view), summary of peaks and movie

        # Uncertainties?


def Slant_stack(st, baz, slows, root=0., coordsys='xy', subsample=True, plot=True, vmin=0.):
    """Vespa process slant stacks, as described in Rost and Thomas (2002)
    st: obspy stream of array data with station coordinates attached
    baz: float or array-like list of floats of backazimuth, if a single float,
        will compute vespagram of slowness for fixed azimuth. If baz is an
        array of values, slowness must be a single float and vice versa.
    slows: float or array-like list of evenly spaced floats of slowness to compute. If a
        single float, will compute vespagram of azimuth for fixed slowness
    root (float): nth root stacking, if n=0, this is a vespagram
    coordsys (str): coordinate system of coordinates of st ('xy' or 'lonlat')
    subsample (bool): if True, will shift timing with subsample accuracy
        by using linear interpolation
    """
    starttime, endtime, sampling_rate, numsamp = checksync(st)
    numsta = len(st)
    
    if isinstance(baz, float):
        fixaz = True
    else:
        fixaz = False
    if isinstance(slows, float):
        fixslow = True
    else:
        fixslow = False
    if not fixaz and not fixslow:
        raise Exception('Must fix either azimuth or slowness')

    geometry = get_geometry(st, coordsys=coordsys)  # returns matrix of x,y,z, each row corresponds to each station in st, relative to center of array (0,0,0)
    tsb = timeshifts(geometry, baz, slows)
    alldata = np.vstack([trace.data for trace in st])
    
    if fixaz:
        vesp = np.zeros((len(slows), numsamp))
        # iterate over slownesses
        for k in range(len(slows)):
            temp = np.zeros(numsamp)
            # get shifts for this slowness in number of samples
            shifts = tsb[:, 0, k] * sampling_rate
            if subsample: # interpolate in shift for subsample accuracy
                for i in range(numsta):
                    tv = np.arange(0., numsamp)
                    if root == 0.: # subtract for subsample add for roll because shifting xaxis right is equivalent to rolling vector back
                        temp += 1./numsta * np.interp(tv - shifts[i], tv, alldata[i, :])
                    else:
                        temp += 1./numsta * np.abs(np.interp(tv - shifts[i], tv, alldata[i, :]))**(1./root) * np.sign(alldata[i, :])
                if root > 0.:
                    temp = np.abs(temp)**root * np.sign(temp)  # take to Nth root and preserve sign

            else: # otherwise round to nearest sample
                shifts = np.round(shifts)
                for i in range(numsta):
                    if root == 0.:
                        temp += 1./numsta * np.roll(alldata[i, :], int(shifts[i]))
                    else:
                        temp += 1./numsta * np.abs(np.roll(alldata[i, :], int(shifts[i])))**(1./root) * np.sign(alldata[i, :])
                if root > 0.:
                    temp = np.abs(temp)**root * np.sign(temp)  # take to Nth root and preserve sign
            vesp[k, :] = temp.copy()

    if fixslow:
        vesp = np.zeros((len(baz), numsamp))
        # iterate over azimuths
        for k in range(len(baz)):
            temp = np.zeros(numsamp)
            # get shifts for this slowness in number of samples
            shifts = tsb[:, k, 0] * sampling_rate
            if subsample: # interpolate in shift for subsample accuracy
                for i in range(numsta):
                    tv = np.arange(0., numsamp)
                    if root == 0.: # subtract for subsample add for roll because shifting xaxis right is equivalent to rolling vector back
                        temp += 1./numsta * np.interp(tv - shifts[i], tv, alldata[i, :])
                    else:
                        temp += 1./numsta * np.abs(np.interp(tv - shifts[i], tv, alldata[i, :]))**(1./root) * np.sign(alldata[i, :])
                if root > 0.:
                    temp = np.abs(temp)**root * np.sign(temp)  # take to Nth root and preserve sign

            else: # otherwise round to nearest sample
                shifts = np.round(shifts)
                for i in range(numsta):
                    if root == 0.:
                        temp += 1./numsta * np.roll(alldata[i, :], int(shifts[i]))
                    else:
                        temp += 1./numsta * np.abs(np.roll(alldata[i, :], int(shifts[i])))**(1./root) * np.sign(alldata[i, :])
                if root > 0.:
                    temp = np.abs(temp)**root * np.sign(temp)  # take to Nth root and preserve sign
            vesp[k, :] = temp.copy()
        
    if plot:
        fig = plt.figure(figsize=(13, 6))
        ax1 = fig.add_subplot(111)
        if fixaz:
            extent = (0., endtime-starttime, np.min(slows), np.max(slows)) # xmin, xmax, ymin, ymax of corners of image (not center of pixel)
        if fixslow:
            extent = (0., endtime-starttime, np.min(baz), np.max(baz))
        ax1.imshow(vesp, extent=extent, aspect='auto', vmin=vmin, cmap=cm.gist_heat_r)
        #labels = ax1.get_yticks()
        #labels = ['%0.2f' % (1./lab) for lab in labels]
        #ax1.set_xticklabels(labels)
        #ax1.set_ylabel('Velocity (km/s)')
        if fixaz:
            ax1.set_ylabel('Slowness (s/km)')
        if fixslow:
            ax1.set_ylabel('Back azimuth (degrees)')
        ax1.set_xlabel('Time (sec)')
        #fig.colorbar(ax1)
        plt.show()

    return vesp


def timeshifts(geometry, baz, slows):
    """Similar to obspy's get_timeshift but does not require equal spacing of gripts
    geometry: output from obspy's get_geometry()
    baz: list of backazimuths in degrees
    slows: list or array of slownesses in s/km
    """
    if isinstance(baz, float):
        baz = [baz]
    if isinstance(slows, float):
        slows = [slows]
    numaz = len(baz)
    numslow = len(slows)
    # unoptimized version for reference
    nstat = len(geometry)  # last index are center coordinates
    
    time_shift_tbl = np.empty((nstat, numaz, numslow), dtype=np.float32)
    for k in xrange(numaz):
        bazkr = math.radians(baz[k])
        for l in xrange(numslow):
            slowl = slows[l]
            # azimuth from center to station k,l (in radians)
            sx = slowl * math.sin(bazkr)
            sy = slowl * math.cos(bazkr)
            time_shift_tbl[:,k,l] = sx * geometry[:, 0] + sy * geometry[:,1]
    return time_shift_tbl
  

def checksync(st, resample=False):
    """Check if all traces in st are synced, if so, return basic info
    
    """
    samprates = [trace.stats.sampling_rate for trace in st]
    if np.mean(samprates) != samprates[0]:
        if resample:
            print('sample rates are not all equal, resampling to lowest sample rate')
            st.resample(np.min(samprates))
        else:
            raise Exception('Sample rates of stream object are not all equal')
    
    stdiff = [st[0].stats.starttime-trace.stats.starttime for trace in st]
    if np.mean(stdiff) != 0.:
        raise Exception('Start times are not uniform')

    stdiff = [st[0].stats.endtime-trace.stats.endtime for trace in st]
    if np.mean(stdiff) != 0.:
        raise Exception('End times are not uniform')
        
    lendiff = [len(st[0])-len(trace) for trace in st]
    if np.mean(lendiff) != 0.:
        raise Exception('Lengths are not equal')
    
    starttime = st[0].stats.starttime
    endtime = st[0].stats.endtime
    sampling_rate = st[0].stats.sampling_rate
    numsamp = len(st[0])

    return starttime, endtime, sampling_rate, numsamp
