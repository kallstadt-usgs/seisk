#!/usr/bin/env python

"""
Tests for reviewData
"""

from reviewData import reviewData as rd
from obspy import UTCDateTime


def test_reviewData():
    """
    Exercises some of the reviewData functions to make sure they don't crash
    """
    starttime = UTCDateTime('2018-06-18T02:34:20')
    endtime = UTCDateTime('2018-06-18T02:37:20')
    st = rd.getdata('IU', 'TEIG,PAYG', '00', 'BHZ', starttime, endtime, savedat=True,
                    filenamepref='Test1_', loadfromfile=True, reloadfile=False)

    event_lat = 14.178
    event_lon = -90.670

    rd.attach_coords_IRIS(st)
    rd.attach_distaz_IRIS(st, event_lat, event_lon)

    fig = rd.recsec(st)

    freqs, amps, fig2 = rd.make_multitaper(st, render=False)

    fig3 = rd.make_spectrogram(st)

    rd.nextpow2(7)

    stacc, stvel = rd.getpeaks(st)

    rd.fourier_spectra(st)


if __name__ == "__main__":
    test_reviewData()
    print('basic tests passed')
