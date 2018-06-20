#!/usr/bin/env python

"""
Tests for sigproc
"""

from sigproc import sigproc as sp
from reviewData import reviewData as rd
from obspy import UTCDateTime


def test_sigproc():
    """
    Exercises some of the reviewData functions to make sure they don't crash
    """
    starttime = UTCDateTime('2018-06-18T02:34:20')
    endtime = UTCDateTime('2018-06-18T02:37:20')
    st = rd.getdata('IU', 'TEIG,PAYG', '00', 'BHZ', starttime, endtime, savedat=True,
                    filenamepref='Test1_', loadfromfile=True, reloadfile=False)
    stnoise = rd.getdata('IU', 'TEIG,PAYG', '00', 'BHZ', starttime - 180., starttime, savedat=True,
                         filenamepref='Test1_', loadfromfile=True, reloadfile=False)

    sp.domfreq(st)
    sp.peakfreq(st)
    sp.meansqfreq(st)
    sp.meansqfreqSN(st, stnoise)
    sp.spectrumSN(st, stnoise)
    sp.powspecSN(st, stnoise)
    sp.multitaper(st)
    sp.multitaperSN(st, stnoise)
    sp.signal_width(st)
    sp.spectrum(st)
    tvec = sp.maketvec(st[0])
    sp.spectrum_manual(st[0].data, tvec)
    sp.unique_list([1, 1, 0])
    sp.xcorrnorm(st[0].resample(20), st[1])
    sp.circshift(st[0], 10)
    sp.subsamplxcorr(st[0], st[1])
    sp.findoutliers([1, 2, 8, 3, 500])


if __name__ == "__main__":
    test_sigproc()
    print('basic sigproc tests passed')
