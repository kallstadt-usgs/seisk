#!/usr/bin/env python

import numpy as np
from obspy.core import Stream
import matplotlib.pyplot as plt
import numpy.ma as ma
import scipy.stats as ss


"""
A collection of signal processing codes for use with obspy and other
miscellaneous functions.
Kate Allstadt - kallstadt@usgs.gov
"""


def domfreq(st, win=None):
    """
    Calculate the dominant frequency of a time series using Douma and Sneider (2006) definition, which is equivalent to a weighted mean, and estimate the variance using the weighted variance formula (amplitudes are weights)
    USAGE
    fd = domfreq(st, win=None)
    INPUTS
    st = obspy stream object, or trace object
    win = tuple of time window in seconds (e.g. win=(3., 20.)) over which to compute dominant frequency, None computes for entire time window
    OUTPUTS
    fd = numpy array of dominant frequencies (Hz)
    """
    st = Stream(st)  # turn into a stream object in case st is a trace
    fd = np.empty(len(st))  # preallocate
    #var = np.empty(len(st))
    #fd2 = np.empty(len(st))
    for i, trace in enumerate(st):
        tvec = maketvec(trace)[:-1]  # Time vector
        vel = trace.data[:-1]
        acc = np.diff(trace.data)*trace.stats.sampling_rate
        if win is not None:
            if win[1] > tvec.max() or win[0] < tvec.min():
                print 'Time window specified not compatible with length of time series'
                return
            vel = vel[(tvec >= win[0]) & (tvec <= win[1])]
            acc = acc[(tvec >= win[0]) & (tvec <= win[1])]
            tvec = tvec[(tvec >= win[0]) & (tvec <= win[1])]
        fd[i] = (np.sqrt(np.trapz(acc**2, tvec)/np.trapz(vel**2, tvec)))/(2*np.pi)
    return fd


def peakfreq(st, win=None):
    """
    Find the frequency with peak amplitude in trace
    USAGE
    fp = peakfreq(trace, win=None)
    INPUTS
    st = obspy stream object, or trace object
    win = tuple of time window in seconds (e.g. win=(3., 20.)) over which to compute peak frequency, None computes for entire time window
    OUTPUTS
    fp = numpy array of peak frequencies (Hz)
    """
    st = Stream(st)  # turn into a stream object in case st is a trace
    fp = np.empty(len(st))  # preallocate
    for i, trace in enumerate(st):
        tvec = maketvec(trace)  # Time vector
        dat = trace.data
        if win is not None:
            if win[1] > tvec.max() or win[0] < tvec.min():
                print 'Time window specified not compatible with length of time series'
                return
            dat = dat[(tvec >= win[0]) & (tvec <= win[1])]
        freqs, amps = spectrum_manual(dat, tvec)
        fp[i] = freqs[amps.argmax()]
        #import pdb; pdb.set_trace()
    return fp


def meansqfreq(st, freqlim=(0, 25), win=None):
    """
    Find the mean squared frequency (Schnabel, 1973)
    Fm = sum(Ci^2*fi)/sum(Ci^2), where Ci is the sqrt of the sum of squares of real and imaginary parts of the fourier coefficients and fi are the frequencies (positive only) - default is to sum between 0 and 25 Hz
        Essentially a weighted mean
    USAGE
    fms, var = meansqfreq(st, freqlim=(0, 25), win=None)
    INPUTS
    st = obspy stream object, or trace object
    freqlim = tuple of frequency limits (Hz) over which to compute mean, default (0, 25)
    win = tuple of time window in seconds (e.g. win=(3., 20.)) over which to compute mean squared frequency, None computes for entire time window in each trace of st
    OUTPUTS
    fms = mean square frequency (Hz)
    var = variance of mean square frequency (Hz**2)
    """
    st = Stream(st)  # turn into a stream object in case st is a trace
    fms = np.empty(len(st))  # preallocate
    var = np.empty(len(st))
    for i, trace in enumerate(st):
        tvec = maketvec(trace)  # Time vector
        dat = trace.data
        if win is not None:
            if win[1] > tvec.max() or win[0] < tvec.min():
                print 'Time window specified not compatible with length of time series'
                return
            dat = dat[(tvec >= win[0]) & (tvec <= win[1])]
        freqs, amps = spectrum_manual(dat, tvec)
        Cis = amps[(freqs >= freqlim[0]) & (freqs <= freqlim[1])]
        freqs1 = freqs[(freqs >= freqlim[0]) & (freqs <= freqlim[1])]
        fms[i] = np.sum(Cis**2*freqs1)/np.sum(Cis**2)
        var[i] = np.sum([(Cis[j]**2*(freq-fms[i])**2) for j, freq in enumerate(freqs1)])/np.sum(Cis**2)
    return fms, var


def meansqfreqSN(st, stnoise, SNrat=1.5, freqlim=(0, 25), win=None):
    """
    Find the mean squared frequency (Schnabel, 1973), but only where SNratio is greater than SNrat
    Fm = sum(Ci^2*fi)/sum(Ci^2), where Ci is the sqrt of the sum of squares of real and imaginary parts of the fourier coefficients and fi are the frequencies (positive only) - default is to sum between 0 and 25 Hz
        Essentially a weighted mean, in this case if SNrat is less than specified, the weight is zero
    USAGE
    fms = meansqfreqSN(st, stnoise, SNrat=1.5, freqlim=(0, 25), win=None)
    INPUTS
    st = obspy stream object, or trace object
    stnoise = obspy stream object from time period just before st (or whatever noise window desired, but needs to have same sample rate)
    freqlim = tuple of frequency limits (Hz) over which to compute mean, default (0, 25)
    win = tuple of time window in seconds (e.g. win=(3., 20.)) over which to compute mean squared frequency, None computes for entire time window in each trace of st
    OUTPUTS
    fms = mean square frequency (Hz)
    var = variance of mean square frequency (Hz**2)
    """
    st = Stream(st)  # turn into a stream object in case st is a trace
    fms = np.empty(len(st))  # preallocate
    var = np.empty(len(st))
    stnoise = Stream(stnoise)
    for i, trace in enumerate(st):
        if trace.stats.sampling_rate != stnoise[i].stats.sampling_rate:
            print 'Signal and noise sample rates are different. Abort!'
            return
        tvec = maketvec(trace)  # Time vector
        dat = trace.data
        if win is not None:
            if win[1] > tvec.max() or win[0] < tvec.min():
                print 'Time window specified not compatible with length of time series'
                return
            dat = dat[(tvec >= win[0]) & (tvec <= win[1])]
            trace = trace.trim(trace.stats.starttime+win[0], trace.stats.starttime+win[1])
        ptvec = maketvec(stnoise[i])
        pdat = stnoise[i].data
        #pdat = pdat[:len(dat)]  # make it the same length as trace
        #ptvec = ptvec[:len(dat)]
        # Find max nfft of the two and use that for both so they line up
        maxnfft = np.max((nextpow2(len(dat)), nextpow2(len(pdat))))
        freqs, amps = spectrum_manual(dat, tvec, nfft=maxnfft)
        pfreqs, pamps = spectrum_manual(pdat, ptvec, nfft=maxnfft)
        idx = (amps/pamps > SNrat) & (freqs >= freqlim[0]) & (freqs <= freqlim[1])  # indices of good values
        Cis = amps[idx]
        freqs1 = freqs[idx]
        fms[i] = np.sum(Cis**2*freqs1)/np.sum(Cis**2)
        var[i] = np.sum([(Cis[j]**2*(freq-fms[i])**2) for j, freq in enumerate(freqs1)])/np.sum(Cis**2)
    return fms, var


def spectrumSN(st, stnoise, SNrat=1.5, win=None):
    """
    Return masked arrays of spectrum, masking where SNratio is greater than SNrat
    USAGE
    freqs, amps, freqmask, ampmask = spectrumSN(st, stnoise, SNrat=1.5, freqlim=(0, 25), win=None)
    INPUTS
    st = obspy stream object, or trace object
    stnoise = obspy stream object from time period just before st (or whatever noise window desired, but needs to have same sample rate)
    win = tuple of time window in seconds (e.g. win=(3., 20.)) over which to compute mean squared frequency, None computes for entire time window in each trace of st
    OUTPUTS
    freqs = list of np.arrays of frequency vectors (Hz)
    amps = list of np.arrays of amplitude vectors
    """
    st = Stream(st)  # turn into a stream object in case st is a trace
    stnoise = Stream(stnoise)
    freqs = []  # preallocate
    amps = []
    ampmask = []
    for i, trace in enumerate(st):
        if trace.stats.sampling_rate != stnoise[i].stats.sampling_rate:
            print 'Signal and noise sample rates are different. Abort!'
            return
        tvec = maketvec(trace)  # Time vector
        dat = trace.data
        ptvec = maketvec(stnoise[i])
        pdat = stnoise[i].data

        if win is not None:
            if win[1] > tvec.max() or win[0] < tvec.min():
                print 'Time window specified not compatible with length of time series'
                return
            dat = dat[(tvec >= win[0]) & (tvec <= win[1])]
            trace = trace.trim(trace.stats.starttime+win[0], trace.stats.starttime+win[1])
        # Find max nfft of the two and use that for both so they line up
        maxnfft = np.max((nextpow2(len(dat)), nextpow2(len(pdat))))
        freqs1, amps1 = spectrum_manual(dat, tvec, nfft=maxnfft)
        pfreqs1, pamps1 = spectrum_manual(pdat, ptvec, nfft=maxnfft)
        idx = (amps1/pamps1 < SNrat)  # good values
        amps.append(amps1)
        freqs.append(freqs1)
        ampmask.append(ma.array(amps1, mask=idx))
    return freqs, amps, ampmask


def powspecSN(st, stnoise, SNrat=1.5, win=None):
    """
    Return masked arrays of power spectrum, masking where SNratio is greater than SNrat
    USAGE
    freqs, amps, ampmask = powspecSN(st, stnoise, SNrat=1.5, freqlim=(0, 25), win=None)
    INPUTS
    st = obspy stream object, or trace object
    stnoise = obspy stream object from time period just before st (or whatever noise window desired, but needs to have same sample rate)
    win = tuple of time window in seconds (e.g. win=(3., 20.)) over which to compute mean squared frequency, None computes for entire time window in each trace of st
    OUTPUTS
    freqs = list of np.arrays of frequency vectors (Hz)
    amps = list of np.arrays of amplitudes
    ampmaks = list of np.arrays of masked amplitudes
    """
    import obspy.signal.spectral_estimation as spec

    st = Stream(st)  # turn into a stream object in case st is a trace
    stnoise = Stream(stnoise)
    freqs = []  # preallocate
    amps = []
    ampmask = []
    for i, trace in enumerate(st):
        if trace.stats.sampling_rate != stnoise[i].stats.sampling_rate:
            print 'Signal and noise sample rates are different. Abort!'
            return
        Fs = trace.stats.sampling_rate  # Time vector
        dat = trace.data
        pFs = stnoise[i].stats.sampling_rate
        pdat = stnoise[i].data
        tvec = maketvec(trace)  # Time vector

        if win is not None:
            if win[1] > tvec.max() or win[0] < tvec.min():
                print 'Time window specified not compatible with length of time series'
                return
            dat = dat[(tvec >= win[0]) & (tvec <= win[1])]
            trace = trace.trim(trace.stats.starttime+win[0], trace.stats.starttime+win[1])
        # Find max nfft of the two and use that for both so they line up
        maxnfft = np.max((nextpow2(len(dat)), nextpow2(len(pdat))))
        Pxx, f = spec.psd(dat, NFFT=maxnfft, Fs=Fs)
        pPxx, pf = spec.psd(pdat, NFFT=maxnfft, Fs=pFs)
        idx = (Pxx/pPxx < SNrat)  # good values
        amps.append(Pxx)
        freqs.append(f)
        ampmask.append(ma.array(Pxx, mask=idx))
    return freqs, amps, ampmask


def signal_width(st):
    pass


def kurtosis(st, winlen, BaillCF=False):
    """
    Compute continuous kurtosis characterisic function using sliding window
    INPUTS
    st - obspy stream or trace
    winlen - window length, in seconds, for continuous kurtosis
    BaillCF - compute characteristic function using methods of Baillard et al. 2014 to estimate first arrival times
    OUTPUTS
    stkurt - st with data replace by either continuous kurtosis or characteristic function from Baillard et al. 2014
    """
    st = Stream(st)
    for trace in st:
        nLEN = int(winlen*trace.stats.sampling_rate)+1
        kurtos = []
        for i in range(len(trace.data)-nLEN+1):
            a = ss.kurtosis(trace.data[i:(i+nLEN)], fisher=False)
            kurtos.append(a)
            if BaillCF is False:
                trace.data = kurtos
            else:
                # Remove all negative slopes
                diffk = np.diff(kurtos)
                F2 = np.zeros(len(kurtos))
                F2[0] = 0.
                for j in range(1, len(kurtos)):
                    if diffk[j-1] < 0:
                        delt = 0.
                    else:
                        delt = 1.
                    F2[j] = F2[j-1] + delt * diffk[j-1]
                b = F2[0]
                a = (F2[-1] - F2[0])/(len(F2)-1)
                F3 = np.zeros(len(kurtos))
                for j in range(1, len(kurtos)):
                    F3[j] = F2[j]-((a*(j-1))+b)
                [M, mintab] = peakdet(F3, (np.max(F3)-np.min(F3))/1000.)
                print (np.max(F3)-np.min(F3))/1000.
                F4 = np.zeros(len(kurtos))
                for j in range(0, len(kurtos)):
                    temp = M[:, 0] - j
                    for k, t in enumerate(temp):
                        if t < 0.:
                            temp[k] = 10.e10
                    index_min = temp.argmin()
                    if F3[j] - M[index_min, 1] < 0.:
                        F4[j] = F3[j] - M[index_min, 1]
                trace.data = F4
    return st


def spectrum(trace, win=None, nfft=None, plot=False, powerspec=False):
    """
    Make amplitude spectrum of trace and plot using rfft (for real inputs, no negative frequencies)
    USAGE
    freqs, amps = spectrum(trace, win=None, plot=False)
    INPUTS
    trace = obspy trace object
    win = tuple of time window in seconds (e.g. win=(3., 20.)) over which to compute amplitude spectrum
    nfft = number of points to use in nfft, default None uses the next power of 2 of length of dat
    plot = True, plot spectrum, False, don't
    powerspec = False for fourier amplitude spectrum, True for power spectrum
    OUTPUTS
    freqs = frequency vector, only positive values
    amps = amplitude vector
    """
    tvec = maketvec(trace)  # Time vector
    dat = trace.data
    if win is not None:
        if win[1] > tvec.max() or win[0] < tvec.min():
            print 'Time window specified not compatible with length of time series'
            return
        dat = dat[(tvec >= win[0]) & (tvec <= win[1])]
        tvec = tvec[(tvec >= win[0]) & (tvec <= win[1])]
    if nfft is None:
        nfft = nextpow2(len(dat))
    if powerspec is False:
        amps = np.abs(np.fft.rfft(dat, n=nfft))
        freqs = np.fft.rfftfreq(nfft, 1/trace.stats.sampling_rate)
    else:
        amps = np.abs(np.fft.fft(dat, n=nfft))**2
        freqs = np.fft.fftfreq(nfft, 1/trace.stats.sampling_rate)
    if plot is True:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        ax.plot(freqs, amps)
        ax.set_xlabel('Frequency (Hz)')
        if powerspec is False:
            plt.title('Amplitude Spectrum')
        else:
            plt.title('Power Spectrum')
        plt.show()
    return freqs, amps


def spectrum_manual(dat, tvec, win=None, nfft=None, plot=False, powerspec=False):
    """
    Make amplitude spectrum of time series and plot using rfft (for real inputs, no negative frequencies)
    USAGE
    freqs, amps = spectrum(trace, win=None, plot=False)
    INPUTS
    dat = obspy trace object
    tvec = time vector for dat
    win = tuple of time window in seconds (e.g. win=(3., 20.)) over which to compute amplitude spectrum
    nfft = number of points to use in nfft, default None uses the next power of 2 of length of dat
    plot = True, plot spectrum, False, don't
    powerspec = False for fourier amplitude spectrum, True for power spectrum
    OUTPUTS
    freqs = frequency vector, only positive values
    amps = amplitude vector
    """
    sample_int = tvec[1]-tvec[0]
    if win is not None:
        if win[1] > tvec.max() or win[0] < tvec.min():
            print 'Time window specified not compatible with length of time series'
            return
        dat = dat[(tvec >= win[0]) & (tvec <= win[1])]
        tvec = tvec[(tvec >= win[0]) & (tvec <= win[1])]
    if nfft is None:
        nfft = nextpow2(len(dat))
    if powerspec is False:
        amps = np.abs(np.fft.rfft(dat, n=nfft))
        freqs = np.fft.rfftfreq(nfft, sample_int)
    else:
        amps = np.abs(np.fft.fft(dat, n=nfft))**2
        freqs = np.fft.fftfreq(nfft, sample_int)
    if plot is True:
        plt.plot(freqs, amps)
        if powerspec is False:
            plt.title('Amplitude Spectrum')
        else:
            plt.title('Power Spectrum')
        plt.show()
    return freqs, amps


def maketvec(trace):
    """
    Make the time vector
    USAGE
    tvec = maketvec(trace)
    INPUTS
    trace = obspy trace
    OUTPUTS
    tvec = numpy array of time vector corresponding to trace.data
    """
    tvec = (np.linspace(0, (len(trace.data)-1)*1/trace.stats.sampling_rate, num=len(trace.data)))
    return tvec


def unique_list(seq):  # make a list only contain unique values and keep their order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def nextpow2(val):
    import math
    temp = math.floor(math.log(val, 2))
    return int(math.pow(2, temp+1))
    pass


def xcorrnorm(tr1, tr2):
    """
    Compute normalized cross correlation of two traces
    maxcor, maxlag, dt = xcorr1x1(tr1, tr2)

    INPUTS
    tr1 - obspy trace1
    tr2 - obspy trace2

    OUTPUTS
    maxcor - value of maximum correlation
    maxlag - lag of maximum correlation (in samples) - this is the number of samples to shift tr2 so it lines up with tr1
    dt - time lag, in seconds

    TODO
    add option to only compute certain lags
    add option to output entire cc function
    """
    from scipy.fftpack import fft, ifft
    if tr1.stats.sampling_rate != tr2.stats.sampling_rate:
        print 'tr1 and tr2 have different sampling rates, abort'
        return
    if len(tr1) != len(tr2):
        print 'tr1 and tr2 are different lengths, abort'
        return
    # pad data to double number of samples to avoid wrap around and pad more to next closest power of 2 for fft
    n2 = nextpow2(len(tr1))

    # make sure data is float
    tr1.data = tr1.data*1.
    tr2.data = tr2.data*1.

    FFT1 = fft(tr1.data, n2)
    norm1 = np.sqrt(np.real(ifft(FFT1*np.conj(FFT1), n2)))
    FFT2 = fft(tr2.data, n2)
    norm2 = np.sqrt(np.real(ifft(FFT2*np.conj(FFT2), n2)))
    cctemp = np.real(ifft(FFT1*np.conj(FFT2), n2))
    cc = cctemp/(norm1[0]*norm2[0])
    M = len(FFT1)
    lags = np.roll(np.linspace(-M/2 + 1, M/2, M, endpoint=True), M/2 + 1).astype(int)
    indx = np.argmax(cc)

    maxcor = cc[indx]
    maxlag = lags[indx]
    dt = 1./tr1.stats.sampling_rate*maxlag
    return maxcor, maxlag, dt


def circshift(tr, ind):
    """
    circular shift of tr by ind samples
    USAGE
    trshift = circshift(tr, ind)

    INPUTS
    tr - trace to shift
    ind - number of samples to shift tr.data
    OUTPUTS
    """
    trshift = tr.copy()
    trshift.data = np.roll(trshift.data, ind)
    trshift.stats.starttime = trshift.stats.starttime + ind*(1./trshift.stats.sampling_rate)
    return trshift


def subsamplxcorr(tr1, tr2, shifts=None):
    """
    Subsample cross correlation - normalized - get subsample shift of tr2 that correlates best with tr1
    maxcor, maxlag = subsamplxcorr(tr1, tr2, shifts=None)
    """
    if tr1.stats.sampling_rate != tr2.stats.sampling_rate:
        print 'tr1 and tr2 have different sampling rates, abort'
        return
    if len(tr1) != len(tr2):
        print 'tr1 and tr2 are different lengths, abort'
        return

    if shifts is None:
        shifts = np.linspace(-0.95, 0.95, 20)
        shifts = [shif for shif in shifts if shif != 0.]  # get rid of zero
    # shift each one and see which one has highest xcorr
    maxcors = []
    for shift in shifts:
        temp = tr2.copy()
        temp.data = fshift(tr2.data, shift)
        maxcor, maxlag, dt = xcorrnorm(tr1, temp)
        maxcors.append(maxcor)
    indx = np.array(maxcors).argmax()
    maxcor = maxcors[indx]
    maxlag = shifts[indx]
    dt = maxlag*(1./tr1.stats.sampling_rate)
    return maxcor, maxlag, dt


def fshift(x, s):
    """
    Fractional circular shift (for use in subsample xcorr)


    Based on code by Francois Bouffard 2005 fbouffar@gel.ulaval.ca -
    Francois' comments about it:
    "FSHIFT circularly shifts the elements of vector x by a (possibly
    non-integer) number of elements s. FSHIFT works by applying a linear
    phase in the spectrum domain and is equivalent to CIRCSHIFT for integer
    values of argument s (to machine precision)."

    """
    from scipy.fftpack import fft, ifft, ifftshift
    N = len(x)
    r = np.floor(N/2)+1
    f = (np.arange(1, N+1)-r)/(N/2)
    p = np.exp(1j*s*np.pi*f)
    y = np.real(ifft(fft(x)*ifftshift(p)))
    return y


def findoutliers(arr, stdequiv=2):

    """
    Identify the indices of outliers of input array arr using the deviations from the median normalized by the median absolute deviation. The threshold for this value is 2.96 for stdequiv = 2 or 4.45 for stdequiv=3 (for the equivalent number of standard deviations).
    USAGE:
    idx = findoutliers(arr, stdequiv=2)

    INPUTS:
    arr = numpy array from which outliers should be identifed
    stdequiv = the number of standard deviation equivalents to use as a threshold

    OUTPUTS:
    idx = indices of outliers
    """
    import statsmodels.robust
    mad = statsmodels.robust.mad(arr)
    z = np.abs((arr-np.median(arr))/mad)
    if stdequiv == 3:
        threshold = 4.45
    else:
        threshold = 2.96
    idx = [i for i in np.arange(len(z)) if z[i] > threshold]
    return idx


def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    PEAKDET Detect peaks in a vector
            [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
            maxima and minima ("peaks") in the vector V.
            MAXTAB and MINTAB consists of two columns. Column 1
            contains indices in V, and column 2 the found values.

            With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
            in MAXTAB and MINTAB are replaced with the corresponding
            X-values.

            A point is considered a maximum peak if it has the maximal
            value, and was preceded (to the left) by a value lower by
            DELTA.

    Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    This function is released to the public domain; Any use is allowed.

    """
    import sys
    from numpy import NaN, Inf, arange, isscalar, asarray, array
    maxtab = []
    mintab = []

    if x is None:
        x = arange(len(v))

    v = asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN

    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)
