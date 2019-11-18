    #!/usr/bin/env python

from obspy import read, Stream
#import matplotlib
#matplotlib.use('TkAgg', warn=False, force=True)
from matplotlib import mlab
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.earthworm import Client as ew_client
from obspy.signal.invsim import simulate_seismometer, corn_freq_2_paz
from obspy import UTCDateTime, Inventory
import numpy as np
import obspy.signal.filter as filte
from obspy.core import AttribDict
import os
from textwrap import wrap
import urllib
from scipy.stats import mode
from sigproc import sigproc
import matplotlib

"""Functions built around obspy for conveniently downloading and interacting with seismic data

Written by kallstadt@usgs.gov
"""
plt.ion()

def getdata(network, station, location, channel, starttime, endtime, attach_response=True,
            clientname='IRIS', savedat=False, folderdat='data',
            filenamepref='Data_', loadfromfile=False, reloadfile=False,
            detrend='demean', merge=True, pad=True, fill_value=0.,
            user=None, password=None):
    """This function grabs data using FDSN webservices

    This is a wrapper around obspy.clients.fdsn that also can automatically
    perform some tasks like saving the data, calling it from memory if already
    saved rather than reloading, and performing basic processing automatically

    Args:
        network (str): seismic network codes, comma separated in a single string
            Example: 'NF,IW,RE,TA,UU'
        station (str): station names, comma separated in a single string
            Example: 'BFR,WOY,TCR,WTM'
        location (str): location codes, comma separated in a single string
            Example: '01,00' or more commonly, just use '*' for all
        channel (str): channels to use, comma separated in a single string
            Example: 'BHZ,BHE,BHN,EHZ'
        starttime: obspy UTCDateTime object of start time
        endtime: obspy UTCDateTime object of end time
        attach_response (bool): attach station response info? True or False
            (default True)
        clientname (str): source of data from FDSN webservices: 'IRIS','NCEDC',
            'GEONET' etc. - see list here http://docs.obspy.org/archive/0.10.2/packages/obspy.fdsn.html
        savedat (bool): True or False, save data locally so it doesn't need to
            be redownloaded to look at it again
        folderdat (str): folder in which to save data, if you save it
        filenamepref (str): - prefix for filename, if you are saving data
        loadfromfile (bool): True or False - if a file from this time period is
            already on the computer, if you say True, it will automatically use
            that file without asking if you want to use it
        reloadfile (bool): if True, will reload locally saved file without asking first
        detrend (str): method to use for detrending in obspy.detrend() before
            merging (if applicable), if None, no detrending will be applied
        merge (bool): if True, will try to merge all traces and will fill with fill_value
        pad (bool): if True, will pad all to be the same length
        fill_value (float or str): fill value for any gaps while merging

    Returns:
        ObsPy stream object containing data that was available for download
            in the same order as input station list

    """
    # strip out white space in inputs
    network = network.replace(' ', '')
    station = station.replace(' ', '')
    location = location.replace(' ', '')
    channel = channel.replace(' ', '')

    #create directory if need be
    if not os.path.exists(folderdat) and savedat is True:
        os.makedirs(folderdat)
    #create file name
    filename = '%s%s_%s' % (filenamepref, starttime.strftime('%Y-%m-%dT%H%M'),
                            endtime.strftime('%Y-%m-%dT%H%M'))
    #see if it exists already
    if os.path.exists(folderdat+'/'+filename):
        if loadfromfile is True:
            choice = 'Y'
        else:
            if reloadfile is False:
                choice = input('file already exists for this time period, enter Y to load from file, N to reload\n')
            else:
                choice = 'N'
    else:
        choice = 'N'
    if choice.upper() == 'Y':
        st_ordered = read(folderdat+'/'+filename, format='PICKLE')
    else:
        try:
            client = FDSN_Client(clientname, user=None, password=None)
            st = client.get_waveforms(network.replace(' ', ''), station.replace(' ', ''),
                                      location.replace(' ', ''), channel.replace(' ', ''),
                                      starttime, endtime, attach_response=True)
            # if any have integer data, turn into float
            for tr in st:
                tr.data = tr.data.astype(float)
            #if detrend is not None:
            #    st.detrend(detrend)
            if merge:
                try:
                    st.merge(fill_value=fill_value)
                except:
                    print('bulk merge failed, trying station by station')
                    st_new = Stream()
                    stationlist = unique_list([trace.stats.station for trace in st])
                    for sta in stationlist:
                        temp = st.select(station=sta)
                        try:
                            temp.merge(fill_value=fill_value)
                            st_new += temp
                        except:
                            # Try resampling
                            sr = [tr.stats.sampling_rate for tr in temp]
                            news = mode(sr)[0][0]
                            temp.resample(news)
                            temp.merge(fill_value=fill_value)
                            st_new += temp
                        #finally:
                        #    print('%s would not merge - deleting it') % (sta,)
                    st = st_new
            if detrend is not None:
                st.detrend(detrend)
            #find min start time
            mint = min([trace.stats.starttime for trace in st])
            if pad:
                st.trim(starttime=mint, pad=True, fill_value=fill_value)
        except Exception as e:
            print(e)
            return
        #make sure it's in the same order as it was originally input
        order = [trace.stats.station for trace in st]
        st_ordered = Stream()
        temp = station.split(',')
        for sta in temp:
            while sta in order:
                indx = order.index(sta)
                st_ordered.append(st[indx])
                st.pop(indx)
                try:
                    order = [trace.stats.station for trace in st]
                except:
                    order = ['', '']
        #save files
        if savedat:
            st_ordered.write(folderdat+'/'+filename, format="PICKLE")
    return st_ordered


def getdata_exact(stations, starttime, endtime, attach_response=True, clientname='IRIS',
                  savedat=False, folderdat='data', filenamepref='Data_',
                  loadfromfile=False, reloadfile=False, detrend='demean',
                  merge=True, pad=True, fill_value=0.,
                  user=None, password=None):
    """This function grabs exact station data using FDSN webservices

    Same as getdata, but only grabs the exact station channel combos specified
    instead of grabbing all. Note this takes longer than getdata.

    Args:
        stations (list): list of tuples in form '[(station,channel,network,loc),]',
            * wildcard can be used for network, channel and loc
        starttime: obspy UTCDateTime object of start time
        endtime: obspy UTCDateTime object of end time
        attach_response (bool): attach station response info? True or False (default True)
        clientname (str): source of data from FDSN webservices: 'IRIS','NCEDC',
            'GEONET' etc. - see list here http://docs.obspy.org/archive/0.10.2/packages/obspy.fdsn.html
        savedat (bool): True or False, save data locally so it doesn't need to
            be redownloaded to look at it again
        folderdat (str): folder in which to save data, if you save it
        filenamepref (str): - prefix for filename, if you are saving data
        loadfromfile (bool): True or False - if a file from this time period is
            already on the computer, if you say True, it will automatically use
            that file without asking if you want to use it
        reloadfile (bool): if True, will reload locally saved file without asking first
        detrend (str): method to use for detrending in obspy.detrend() before
            merging (if applicable), if None, no detrending will be applied
        merge (bool): if True, will try to merge all traces and will fill with fill_value
        pad (bool): if True, will pad all to be the same length
        fill_value (float or str): fill value for any gaps while merging

    Returns:
        ObsPy stream object containing data that was available for download
            in the same order as input station list

    """
    #create directory if need be
    if not os.path.exists(folderdat) and savedat is True:
        os.makedirs(folderdat)
    #create file name
    filename = '%s%s_%s' % (filenamepref, starttime.strftime('%Y-%m-%dT%H%M'),
                            endtime.strftime('%Y-%m-%dT%H%M'))
    #see if it exists already
    if os.path.exists(folderdat+'/'+filename):
        if loadfromfile is True:
            choice = 'Y'
        else:
            if reloadfile is False:
                choice = input('file already exists for this time period, enter Y to load from file, N to reload\n')
            else:
                choice = 'N'
    else:
        choice = 'N'
    if choice.upper() == 'Y':
        st = read(folderdat+'/'+filename, format='PICKLE')
    else:
        st = Stream()
        try:
            client = FDSN_Client(clientname, user=user, password=password)
            for statup in stations:
                try:
                    sttemp = client.get_waveforms(statup[2], statup[0], statup[3], statup[1],
                                                  starttime, endtime, attach_response=True)
                                # if any have integer data, turn into float
                    for tr in sttemp:
                        tr.data = tr.data.astype(float)
                    if detrend is not None:
                        sttemp.detrend(detrend)
                    if merge:
                        try:
                            sttemp.merge(fill_value=fill_value)
                        except:
                            print('bulk merge failed, trying station by station')
                            st_new = Stream()
                            stationlist = unique_list([trace.stats.station for trace in sttemp])
                            for sta in stationlist:
                                temp = sttemp.select(station=sta)
                                try:
                                    temp.merge(fill_value=fill_value)
                                    st_new += temp
                                except:
                                    # Try resampling
                                    sr = [tr.stats.sampling_rate for tr in temp]
                                    news = mode(sr)[0][0]
                                    temp.resample(news)
                                    temp.merge(fill_value=fill_value)
                                    st_new += temp
                                #finally:
                                #    print('%s would not merge - deleting it') % (sta,)
                            sttemp = st_new
                    if detrend is not None:
                        sttemp.detrend(detrend)
                    st += sttemp.copy()
                except Exception as e:
                    print(e)
                    print((('failed to grab data from %s, moving on') % (statup,)))
            if detrend is not None:
                st.detrend('demean')
            #find min start time
            mint = min([trace.stats.starttime for trace in st])
            if pad:
                st.trim(starttime=mint, pad=True, fill_value=fill_value)
        except Exception as e:
            print(e)
            return

        #save files
        if savedat:
            st.write(folderdat+'/'+filename, format="PICKLE")
    return st


def getdata_winston(stations, starttime, endtime, ewclientname, port, chanuse='*',
                    attach_response=True, respclientname='IRIS', savedat=False,
                    folderdat='data', filenamepref='Data_', loadfromfile=False,
                    reloadfile=False, detrend='demean', merge=True, pad=True,
                    fill_value=0.):
    """Get data from winston waveserver

    Same as getdata, but only grabs the exact station channel combos specified
    instead of grabbing all. Note this takes longer than getdata.

    Args:
        stations (list): list of tuples in form '[(station,channel,network,loc),]',
            * wildcard can be used for network, channel and loc
        starttime: obspy UTCDateTime object of start time
        endtime: obspy UTCDateTime object of end time
        ewclientname: Earthworm client name (may be an ip address)
        port: Earthworm port number
        chanuse (str): which channels to keep in comma separate single string,
            e.g. 'EHZ,BHZ', if all are ok, insert * (default)
        attach_response (bool): Try to attach station response info?
            True or False (default True), will try to get response info from
            FDSN client specified by respclientname
        respclientname (str): source of data from FDSN webservices: 'IRIS','NCEDC',
            'GEONET' etc. - see list here http://docs.obspy.org/archive/0.10.2/packages/obspy.fdsn.html
        savedat (bool): True or False, save data locally so it doesn't need to
            be redownloaded to look at it again
        folderdat (str): folder in which to save data, if you save it
        filenamepref (str): - prefix for filename, if you are saving data
        loadfromfile (bool): True or False - if a file from this time period is
            already on the computer, if you say True, it will automatically use
            that file without asking if you want to use it
        reloadfile (bool): if True, will reload locally saved file without asking first
        detrend (str): method to use for detrending in obspy.detrend() before
            merging (if applicable), if None, no detrending will be applied
        merge (bool): if True, will try to merge all traces and will fill with fill_value
        pad (bool): if True, will pad all to be the same length
        fill_value (float or str): fill value for any gaps while merging

    Returns:
        ObsPy stream object containing data that was available for download
            in the same order as input station list

    """
    #create directory if need be
    if not os.path.exists(folderdat):
        os.makedirs(folderdat)
    #create file name
    filename = '%s%s_%s' % (filenamepref, starttime.strftime('%Y-%m-%dT%H%M'),
                            endtime.strftime('%Y-%m-%dT%H%M'))
    #see if it exists already
    if os.path.exists(folderdat+'/'+filename):
        if loadfromfile is True:
            choice = 'Y'
        else:
            if reloadfile is False:
                choice = input('file already exists for this time period, enter Y to load from file, N to reload\n')
            else:
                choice = 'N'
    else:
        choice = 'N'
    if choice.upper() == 'Y':
        st = read(folderdat+'/'+filename, format="PICKLE")
    else:
        #loop through all stations
        flag = 0
        for sta in stations:
            #get station, channel, network and do some logic
            if sta[1] in chanuse or chanuse == '*':
                try:
                    client = ew_client(ewclientname, port)
                    temp = client.get_waveforms(sta[2], sta[0], '', sta[1], starttime, endtime)
                    for tr in temp:
                        tr.data = tr.data.astype(float)
                    if detrend is not None:
                        temp.detrend(detrend)
                    if merge:
                        temp.merge(fill_value=fill_value)
                    if flag == 0:
                        st = temp
                        flag = 1
                    else:
                        st += temp
                except Exception as e:
                    print(e)
                    print(('No data available from %s.%s.%s' % (sta)))
        #find min start time
        if 'st' in locals():
            if len(st) != 0:  # make sure st isn't empty
                mint = min([trace.stats.starttime for trace in st])
                if pad:
                    st.trim(starttime=mint, pad=True, fill_value=fill_value)
                if attach_response is True:
                    client = FDSN_Client(respclientname)  # try to get responses and attach them
                    try:
                        client._attach_responses(st)
                    except:
                        print('could not attach response info, station correction will not work')
                if savedat:
                    st.write(folderdat+'/'+filename, format='PICKLE')
                print(st)
                return st
            else:
                print('No data returned')
        else:
            print('No data returned')


def getdata_filenames(filenames, chanuse='*', starttime=None, endtime=None,
                      attach_response=False, respclientname='IRIS', savedat=False,
                      folderdat='data', filenamepref='Data_', loadfromfile=False,
                      reloadfile=False, detrend='demean', merge=True, pad=True,
                      fill_value=0.):
    """Read in sac, mseed or other locally stored files

    Note that this function is only more useful than obspy.read in that it can
    save subsets of larger files for easier access and can attach response
    information (if available)

    Args:
        filenames (list): list of full filenames (e.g. glob output)
            * wildcard can be used for network, channel and loc
        chanuse (str): which channels to keep in comma separate single string,
            e.g. 'EHZ,BHZ', if all are ok, insert * (default)
        starttime: obspy UTCDateTime object of start time
        endtime: obspy UTCDateTime object of end time

        attach_response (bool): Try to attach station response info?
            True or False (default True), will try to get response info from
            FDSN client specified by respclientname
        respclientname (str): source of data from FDSN webservices: 'IRIS','NCEDC',
            'GEONET' etc. - see list here http://docs.obspy.org/archive/0.10.2/packages/obspy.fdsn.html
        savedat (bool): True or False, save data locally so it doesn't need to
            be redownloaded to look at it again
        folderdat (str): folder in which to save data, if you save it
        filenamepref (str): - prefix for filename, if you are saving data
        loadfromfile (bool): True or False - if a file from this time period is
            already on the computer, if you say True, it will automatically use
            that file without asking if you want to use it
        reloadfile (bool): if True, will reload locally saved file without asking first
        detrend (str): method to use for detrending in obspy.detrend() before
            merging (if applicable), if None, no detrending will be applied
        merge (bool): if True, will try to merge all traces and will fill with fill_value
        pad (bool): if True, will pad all to be the same length
        fill_value (float or str): fill value for any gaps while merging

    Returns:
        ObsPy stream object containing data that was available for download
            in the same order as input station list

    """
    #create directory if need be
    if not os.path.exists(folderdat):
        os.makedirs(folderdat)
    #create file name
    if starttime is None or endtime is None:
        filename = filenamepref
    else:
        #filename = filenamepref+str(starttime)+str(endtime)
        filename = '%s%s_%s' % (filenamepref, starttime.strftime('%Y-%m-%dT%H%M'),
                                endtime.strftime('%Y-%m-%dT%H%M'))
    #see if it exists already
    if os.path.exists(folderdat+'/'+filename):
        if loadfromfile is True:
            choice = 'Y'
        else:
            if reloadfile is False:
                choice = input('file already exists for this time period, \
                                enter Y to load from file, N to reload\n')
            else:
                choice = 'N'
    else:
        choice = 'N'
    if choice.upper() == 'Y':
        st = read(folderdat+'/'+filename, format="PICKLE")
    else:
        st = Stream()
        for file1 in filenames:
            try:
                temp = read(file1)
                if temp[0].stats.channel in chanuse or chanuse == '*':
                    if attach_response is True:
                        client = FDSN_Client(respclientname)
                        try:
                            client._attach_responses(temp)
                            st += temp
                        except:
                            print(('could not attach response info for %s, \
                            station correction will not work' % temp.stats.station))
                    else:
                        st += temp
            except Exception as e:
                print(e)
                print(('could not read %s, skipping to next file name' % file1))
        for tr in st:
            tr.data = tr.data.astype(float)
        if detrend is not None:
            st.detrend(detrend)
        if merge:
            try:
                st.merge(fill_value=fill_value)
            except:
                print('bulk merge failed, trying station by station')
                st_new = Stream()
                stationlist = unique_list([trace.stats.station for trace in st])
                for sta in stationlist:
                    temp = st.select(station=sta)
                    for tr in temp:
                        tr.data = tr.data.astype(float)
                    try:
                        temp.merge(fill_value=fill_value)
                        st_new += temp
                    except:
                        # Try resampling
                        sr = [tr.stats.sampling_rate for tr in temp]
                        news = mode(sr)[0][0]
                        temp.resample(news)
                        temp.merge(fill_value=fill_value)
                        st_new += temp
                    finally:
                        print((('%s would not merge - deleting it') % (sta,)))
                st = st_new
        if starttime or endtime:
            if pad:
                st.trim(starttime=starttime, endtime=endtime, pad=True,
                        fill_value=fill_value)
        else:  # find min start time and trim all to same point
            mint = min([trace.stats.starttime for trace in st])
            if pad:
                st.trim(starttime=mint)  # ,pad = True, fill_value = 0)
        if savedat:  # save file if so choose
            st.write(folderdat+'/'+filename, format='PICKLE')
    return st


def getepidata(event_lat, event_lon, event_time, tstart=-5., tend=200.,
               minradiuskm=0., maxradiuskm=20., chanuse='*', location='*',
               clientnames=['IRIS'], attach_response=True, savedat=False,
               folderdat='data', filenamepref='Data_', loadfromfile=False,
               reloadfile=False, detrend=None, merge=False, pad=True,
               fill_value=0.):
    """Pull existing data within a certain radius of the epicenter

    Or from any lat/lon coordinates. Also attaches station coordinates to data

    Args:
        event_lat (float): latitude in decimal degrees
        event_lon (float): longitude in decimal degrees
        event_time: obspy UTCDateTime object of event time
        tstart (float): Number of seconds after the event time to use as
            starttime for data (negative numbers will grab data from before)
        tend (float): Number of seconds after the event time to use as
            starttime for data
        minradius (float): minimum radius (km) to search
        maxradius (float): maximum radius (km) to search
        chanuse (str): 'strong motion' to get all strong motion channels
            (excluding low sample rate ones), 'broadband' to get all broadband
            instruments, 'short period' for all short period channels,
            otherwise a single line of comma separated channel codes,
            * wildcards are okay, e.g. channels = '*N*,*L*'
        location (str): comma separated list of location codes allowed,
            or '*' for all location codes
        clientnames (list): list of FDSN clients to search over, e.g. ['IRIS']
            see list of options here http://docs.obspy.org/archive/0.10.2/packages/obspy.fdsn.html
        attach_response (bool): Attach station response info (if available)
        savedat (bool): True or False, save data locally so it doesn't need to
            be redownloaded to look at it again
        folderdat (str): folder in which to save data, if you save it
        filenamepref (str): - prefix for filename, if you are saving data
        loadfromfile (bool): True or False - if a file from this time period is
            already on the computer, if you say True, it will automatically use
            that file without asking if you want to use it
        reloadfile (bool): if True, will reload locally saved file without asking first
        detrend (str): method to use for detrending in obspy.detrend() before
            merging (if applicable), if None, no detrending will be applied
        merge (bool): if True, will try to merge all traces and will fill with fill_value
        pad (bool): if True, will pad all to be the same length
        fill_value (float or str): fill value for any gaps while merging

    Returns:
        ObsPy stream object containing data that was available for download
            within the requested area in order of increasing distance
            from the sourcei

    """
    event_time = UTCDateTime(event_time)
    st_all = Stream()

    # If clientnames is not a list, make it a list
    if type(clientnames) != list:
        clientnames = [clientnames]

    for source in clientnames:
        client = FDSN_Client(source)

        if chanuse.lower() == 'strong motion':
            chanuse = 'EN*,HN*,BN*,EL*,HL*,BL*'
        elif chanuse.lower() == 'broadband':
            chanuse = 'BH*,HH*'
        elif chanuse.lower() == 'short period':
            chanuse = 'EH*'
        else:
            chanuse = chanuse.replace(' ', '')  # Get rid of spaces

        t1 = UTCDateTime(event_time) + tstart
        t2 = UTCDateTime(event_time) + tend

        inventory = client.get_stations(latitude=event_lat, longitude=event_lon,
                                        minradius=minradiuskm/111.32,
                                        maxradius=maxradiuskm/111.32,
                                        channel=chanuse, level='channel')
                                        #startbefore=t1, endafter=t2)
        temp = inventory.get_contents()
        netnames = temp['networks']
        stas = temp['stations']
        stanames = [n.split('.')[1].split()[0] for n in stas]

        st = getdata(','.join(unique_list(netnames)), ','.join(unique_list(stanames)),
                     location, chanuse, t1, t2, attach_response=attach_response,
                     clientname=source, savedat=savedat, folderdat=folderdat,
                     filenamepref=filenamepref, loadfromfile=loadfromfile,
                     reloadfile=reloadfile)

        if st is None:
            print('No data returned')
            return

        if detrend is not None:
            st.detrend(detrend)
        if merge:
            try:
                st.merge(fill_value=fill_value)
            except:
                print('bulk merge failed, trying station by station')
                st_new = Stream()
                stationlist = unique_list([trace.stats.station for trace in st])
                for sta in stationlist:
                    temp = st.select(station=sta)
                    for tr in temp:
                        tr.data = tr.data.astype(float)
                    try:
                        temp.merge(fill_value=fill_value)
                        st_new += temp
                    except:
                        # Try resampling
                        sr = [tr.stats.sampling_rate for tr in temp]
                        news = mode(sr)[0][0]
                        temp.resample(news)
                        temp.merge(fill_value=fill_value)
                        st_new += temp
                    finally:
                        print((('%s would not merge - deleting it') % (sta,)))
                st = st_new
        for trace in st:
            try:
                coord = inventory.get_coordinates(trace.id)
                trace.stats.coordinates = AttribDict({'latitude': coord['latitude'],
                                                      'longitude': coord['longitude'],
                                                      'elevation': coord['elevation']})
            except:
                print(('Could not attach coordinates for %s' % trace.id))

        st_all += st

    return st_all


def recsec(st, norm=True, xlim=None, ylim=None, scalfact=1., tscale='relative',
           update=False, fighandle=None, indfirst=0, maxtraces=10, textbox=False,
           textline=['>', '>', '>', '>', '>'], menu=None, quickdraw=True,
           labelquickdraw=True, processing=None, figsize=None, colors=None,
           labelsize=12., addscale=False, unitlabel=None, convert=1.,
           scaleperc=0.9, vlines=None, vlinestyle='--', vlinecolor='k',
           picktimes=None, pad=False, fill_value=0., xonly=False, samezero=False, grid=True,
           title=True, labellist=None, labelloc='left', block=False):
    """Plot record section of data from an obspy stream

    Note that a lot of the functionality is built in for use by
    InteractivePlot and is not useful for standalone use.

    Args:
        st: obspy stream to plot
        norm (bool): True or False, normalize each trace by its maximum
        xlim (tup): tuple of axes limits e.g. (0,100),
            None uses default axis limits
        ylim (tup): tuple of axes limits e.g. (0,100),
            None uses default axis limits
        tscale (str): 'relative' uses seconds relative to start time for x axis
            labels, 'absolute' uses absolute time.
        scalfact (float): scaling factor for plotting, 1 is default,
            2 would be double the amplitudes, 0.5 halves them etc.
        update (bool): False if this is a new plot (False), True if it's an update
        fighandle: figure handle of plot to update. New handle name is given if
            figure isn't already open.
        indfirst (int): index of first trace of st to plot
        maxtraces (int): maximum number of traces to plot
        textbox (bool): Whether to show the textbox (required for InteractivePlot)
        textline (list): List of lines of text that will be printed in text box.
        menu (str) = Text of help menu to print, None for no help menu
        quickdraw (bool): Whether to use obsPy's minmax plot method to make
            plotting much faster. Only applies if there are more than 30
            samples per pixel on the plot.
        labelquickdraw (bool): Whether to label the plot with a qd in the upper
            right corner, indicating quickdraw is on.
        processing (bool): Whether to show processing history saved in stats,
            otherwise None
        figsize (tuple): tuple of figure size in inches e.g., (10, 10)
            is 10x10inches, (width, height)
        colors (list) = list of colors to apply to each trace or single color
            to apply to all traces. If None, will use defaults.
        labelsize (float): font size for labels
        addscale (bool): add a scale bar to each trace showing the amplitudes
        unitlabel (str) = String of label for scale bar, if None, won't label.
        convert (float): Factor to multipy by to convert data amplitudes into
            unitlabel units (e.g. to convert m/s to mm/s, convert=10**3)
        scalebarperc (float): proportion of total axis width over from left
            edge of plot to place scale bar (between 0 and 1)
        vlines: None, or array of obspy UTCDateTimes where vertical lines
            should be placed (e.g. to show event start times)
        vlinestyle (str): vline style shortcut e.g. '--'
        vlinecolor (str): vline color shortcut e.g. 'k'
        picktimes: None or array of obspy UTCDateTimes of pick times to plot on each
            trace corresponding to and in the same order as the stream
        pad (bool): whether to pad traces so they are all the same length
        fill_value (float or int): fill value to pad with, if None, will create
            masked array, which may present challenges to further processing the data
        xonly (bool): if True, only bottom x-axis will be visible
        samezero (bool): if True, will line all traces up so the first sample
            is at zero regardless of actual start time.
        grid (bool): if True, grid on
        title (bool): if True, title will be added
        labellist (list): if None, will create own labels, otherwise list of
            labels corresponding to each trace (same order as st)
        labelloc (str): 'left' will be left of traces, 'below' will be below
            each trace at left, 'above' will be above each trace at left
        block (bool): if True, will block code from continuing until figure is closed

    Returns:
        handle of figure

    """
    # Make sure not to modify the original
    st = st.copy()

    if samezero:
        for tr in st:
            tr.stats.starttime = UTCDateTime(1970, 1, 1, 0, 0, 0)
    try:
        maxtraces = min(len(st), maxtraces)
        maxtraces = int(maxtraces)
    except Exception:
        print('st is empty')
        return
    #set up color cycling
    if type(colors) == str:
        rep = len(st)*[colors]
    elif colors is None or len(colors) != len(st):
        colors_ = ['r', 'b', 'g', 'm', 'k', 'c']
        rep = int(np.ceil(float(len(st))/float(len(colors_))))*colors_
        rep = rep[0:len(st)]
    else:
        rep = colors

    if fighandle is not None:
        #try:
        fig = fighandle
        if textbox is True:
            axbox, ax = fig.get_axes()
            boxes = ax.texts
            for b in boxes:
                b.remove()
        else:
            ax = fig.get_axes()
        #except:
        #    print('need to define fighandle correctly to update current plot, creating new figure')
        #    update = False

    if update is True and fighandle is None:
        print('Cannot update without specifying a figure handle, creating new figure')
        update = False

    if update is False:
        if fighandle is not None:
            ax.clear()
            if textbox is True:
                axbox.clear()
        else:
            if figsize is None:
                fig = plt.figure(figsize=(12, min(10, 3*len(st))))
            else:
                fig = plt.figure(figsize=figsize)
            if textbox is True:
                axbox = fig.add_axes([0.2, 0.05, 0.75, 0.1])
                ax = fig.add_axes([0.2, 0.23, 0.75, 0.72])  # left bottom width height
            else:
                ax = fig.add_axes([0.23, 0.13, 0.72, 0.78])

    if labellist is None:
        labels = []
    else:
        labels = labellist
    yticks1 = []
    maxy = 0
    miny = 0
    fig.stationsy = {}  # create empty dictionary to save which station is at which y-value

    # Pad traces so they have same start and end times
    tmin = min([trace.stats.starttime for trace in st])
    tmax = max([trace.stats.endtime for trace in st])

    if pad:
        st = st.trim(tmin, tmax, pad=True, fill_value=fill_value)

    if xlim is None:
        xlim = [0, tmax-tmin]

    # get x location of scale bar to apply to all
    xerrloc = tmin + ((tmax-tmin) * scaleperc)

    avgmax = np.median(np.absolute(st.max()))  # parameter used for scaling traces relative to each other
    i = 0
    missing = 0

    flag = 0
    for st1, color1 in zip(st[indfirst:indfirst+maxtraces], rep[indfirst:indfirst+maxtraces]):
        dat = st1.data
        samprate = st1.stats.sampling_rate
        tdiff = st1.stats.starttime - tmin
        x_width = xlim[1] - xlim[0]
        pixel_length = int(np.ceil((x_width * st1.stats.sampling_rate + 1) / (fig.get_size_inches()[1]*fig.dpi)))  # Number of samples per pixel on plot
        if quickdraw is True and pixel_length > 30:  # x_width*st1.stats.sampling_rate > 10000:
            pixel_count = int(len(dat) // pixel_length)
            remaining_samples = int(len(st1.data) % pixel_length)
            #remaining_seconds = remaining_samples / st1.sampling_rate
            if remaining_samples:
                dat = dat[:-remaining_samples]
            dat = dat.reshape(pixel_count, pixel_length)
            min_ = dat.min(axis=1)
            max_ = dat.max(axis=1)
            # Calculate extreme_values and put them into new array.
            extreme_values = np.empty((pixel_count, 2), dtype=np.float)
            extreme_values[:, 0] = min_
            extreme_values[:, 1] = max_
            x_values = np.linspace(0, (extreme_values.shape[0]-1)*(pixel_length/st1.stats.sampling_rate),
                                   num=extreme_values.shape[0])
            tvec = np.repeat(x_values, 2) + tdiff
            dat = extreme_values.flatten()
            mask = ((tvec > xlim[0]) & (tvec < xlim[1]))
            flag = 1
        else:
            tvec = (np.linspace(0, (len(dat)-1)*1/samprate, num=len(dat))) + tdiff
            mask = ((tvec > xlim[0]) & (tvec < xlim[1]))
            dat = st1.data
        try:
            temp = st1.stats.station+'.'+st1.stats.channel+'.'+st1.stats.location+'.'+st1.stats.network
            staname = ('%s - %2.1f km') % (temp, st1.stats.rdist)
        except:
            staname = st1.stats.station+'.'+st1.stats.channel+'.'+st1.stats.location+'.'+st1.stats.network

        # Plot data
        try:
            if norm is True:
                yloc = -2.*i
                maxval = max(np.absolute(dat[mask]))
                dat = scalfact*dat/maxval
                if update is False:
                    ax.plot(tvec[mask], np.add(dat[mask], yloc), color1)
                elif update is True:
                    ax.lines[i].set_data(tvec[mask], np.add(dat[mask], yloc))
                    ax.lines[i].set_color(color1)
                tempmax = np.add(dat[mask], yloc).max()
                tempmin = np.add(dat[mask], yloc).min()
                if tempmax > maxy:
                    maxy = tempmax
                if tempmin < miny:
                    miny = tempmin
                yticks1.append(yloc)
                if labellist is None:
                    labels.append(staname)
                fig.stationsy[yloc] = staname
                if addscale:
                    ax.errorbar(xerrloc, yloc, yerr=0.5, color='0.5')
                    # place labels
                    label1 = '  -%.1E' % (0.5*maxval * convert)
                    label1 = label1.lower().replace('+', '').replace('e0', 'e')
                    label2 = '   %.1E' % (0.5*maxval * convert)
                    label2 = label2.lower().replace('+', '').replace('e0', 'e')
                    ax.text(xerrloc, yloc + 0.5, label2, color='0.5', fontsize=11,
                            horizontalalignment='left', verticalalignment='center')
                    ax.text(xerrloc, yloc - 0.5, label1, color='0.5', fontsize=11,
                            horizontalalignment='left', verticalalignment='bottom')

            else:
                yloc = -2.*avgmax*i
                dat = scalfact*dat
                maxval = max(np.absolute(dat[mask]))
                if update is False:
                    ax.plot(tvec[mask], np.add(dat[mask], yloc), color1)
                elif update is True:
                    ax.lines[i].set_data(tvec[mask], np.add(dat[mask], yloc))
                    ax.lines[i].set_color(color1)
                tempmax = np.add(dat, yloc).max()
                tempmin = np.add(dat, yloc).min()
                if tempmax > maxy:
                    maxy = tempmax
                if tempmin < miny:
                    miny = tempmin
                yticks1.append(yloc)
                if labellist is None:
                    labels.append(staname)
                fig.stationsy[yloc] = staname
                if addscale:
                    if i == 0:
                        ax.errorbar(xerrloc, yloc, yerr=0.5*maxval, color='0.5')
                        # place labels
                        label1 = '  -%.1E' % (0.5*maxval/scalfact * convert)
                        label1 = label1.lower().replace('+', '').replace('e0', 'e').replace('e-0', 'e-')
                        label2 = '   %.1E' % (0.5*maxval/scalfact * convert)
                        label2 = label2.lower().replace('+', '').replace('e0', 'e').replace('e-0', 'e-')
                        ax.text(xerrloc, yloc + 0.5*maxval, label2, color='0.5', fontsize=11,
                                horizontalalignment='left', verticalalignment='center')
                        ax.text(xerrloc, yloc - 0.5*maxval, label1, color='0.5', fontsize=11,
                                horizontalalignment='left', verticalalignment='bottom')
        except:
            missing += 1
        i += 1
    if missing == len(st):
        print('nothing to see over here, go back')
    plt.tick_params(left=False, right=False)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.autoscale(enable=True, axis='y', tight=True)

    plt.yticks(yticks1)
    if labelloc == 'below':
        ax.set_yticklabels(labels, fontsize=labelsize, position=(0, 0), ha='left', va='top')
    elif labelloc == 'above':
        ax.set_yticklabels(labels, fontsize=labelsize, position=(0, 0), ha='left', va='bottom')
    else:
        ax.set_yticklabels(labels, fontsize=labelsize)
    ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ax.set_ylim(miny, maxy)
    if grid:
        plt.grid(True)

    mcs = ('%.2f' % (tmin.microsecond/10.**6)).replace('0.', '')
    timeprint = tmin.strftime('%Y-%m-%dT%H:%M:%S.') + mcs

    if samezero:
        sttm = ''
    else:
        sttm = 'Start time: %s (UTC) - ' % timeprint
    if title:
        if len(st) <= maxtraces:
            title1 = '%s%i traces' % (sttm, len(st))
        else:
            title1 = '%sTraces %i through %i of %i' % (sttm, indfirst+1, indfirst+maxtraces, len(st))
        if addscale is True and unitlabel is not None:
            title1 = title1 + ' - Units = %s' % unitlabel
        if norm:
            title1 = title1 + ' - normalized to maxes'
        else:
            title1 = title1 + ' - same amplitudes'
        plt.title(title1, fontsize=labelsize + 1.)

    if hasattr(st[0].stats, 'processing'):
        proc = wrap('PROCESSING HISTORY: %s' % ' - '.join(st[0].stats.processing[0:]), 50)
    else:
        proc = ' '
    props = dict(facecolor='wheat', alpha=0.3)
    # print processing summary if desired
    if processing is not None:
        #ax.text(0.02, 0.97, '\n'.join(proc), transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
        ax.annotate('\n'.join(proc), xy=(0.02, 0.97), xycoords='axes fraction',
                    fontsize=8, verticalalignment='top', bbox=props)
    #print help menu, if desired
    if menu is not None:
        #ax.text(0.02,0.8,menu,transform=ax.transAxes,fontsize=8,verticalalignment='top',bbox=props)
        ax.annotate(menu, xy=(0.02, 0.8), xycoords='axes fraction', fontsize=8,
                    verticalalignment='top', bbox=props)

    #print command line box
    if textbox is True:
        axbox.clear()
        axbox.set_yticks([])
        axbox.set_xticks([])
        axbox.text(0.01, 0.99, '\n'.join(textline[-5:]), transform=axbox.transAxes, fontsize=12, verticalalignment='top')

    if xonly:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)

    if vlines is not None:
        for vlin in vlines:
            ax.axvline(vlin - tmin, linestyle=vlinestyle, color=vlinecolor)

    if picktimes is not None:
        try:  # Get the right picktimes for current display
            picktimes2 = picktimes[indfirst:indfirst+maxtraces]
            # Get avg distance between lines
            ht = np.abs(np.mean(np.diff(yticks1)))/2.
            for pick, yloc1 in zip(picktimes2, yticks1):
                if isinstance(pick, UTCDateTime):
                    xloc1 = pick - tmin
                else: # must be in seconds from record start time
                    xloc1 = pick
                ax.plot([xloc1, xloc1], [yloc1-ht, yloc1+ht], color='0.0013',
                        linestyle='-', linewidth=1.65, zorder=1.e10)
                #ax.vlines(x=xloc1, ymin=yloc1-ht, ymax=yloc1+ht, color='0.0013',
                #          linestyle='-', linewidth=1.8, zorder=1.e10)
                #ax.errorbar(xloc1, yloc1, yerr=ht, color='k', linestyle='-', zorder=1.e10)
        except:
            print('picktimes array needs to be same length as input stream, skipping')


    props1 = dict(facecolor='white', alpha=1)
    if labelquickdraw:
        if quickdraw is True and flag == 1:
            ax.annotate('qd on ', xy=(0.9, 0.95), xycoords='axes fraction', bbox=props1)
        elif quickdraw is True and flag == 0:
            ax.annotate('qd off', xy=(0.9, 0.95), xycoords='axes fraction', bbox=props1)

    if tscale == 'relative':
        plt.xlabel('Time (sec)', fontsize=12)
    else:
        # Change yaxis labels to absolute time
        seclabels = ax.get_xticks()
        newlabs = []
        for sec in seclabels:
            value1 = tmin + sec
            if tmax-tmin > 3600.*24.:  # greater than a day, include day but not second
                newlabs.append(value1.strftime('%d %b-%H:%M'))
                rotation = 5
            elif tmax-tmin > 3600.:  # greater than an hour, show full seconds
                newlabs.append(value1.strftime('%d %b-%H:%M:%S'))
                rotation = 8
            else:
                mcs = ('%.2f' % (value1.microsecond/10.**6)).replace('0.', '')
                newlabs.append(value1.strftime('%H:%M:%S.') + mcs)
                rotation = 0
        ax.set_xticklabels(newlabs, fontsize=labelsize, ha="center", rotation=rotation)  # , rotation=20)

    if update is False:
        plt.show(block=block)
    elif update is True:
        ax.figure.canvas.draw()
        if textbox is True:
            axbox.figure.canvas.draw()

    return fig


def make_multitaper(st, number_of_tapers=None, time_bandwidth=4., sine=False, recsec=False, labelsize=12,
                    logx=False, logy=False, xunits='Hz', xlim=None, ylim=None, yunits=None, colors1=None, render=True,
                    detrend='demean'):
    """
    Plot multitaper spectra of signals in st, equivalent to power spectral density

    # TO DO add statistics and optional_output options

    :param st: obspy stream or trace containing data to plot
    :param number_of_tapers: Number of tapers to use, Defaults to int(2*time_bandwidth) - 1.
     This is maximum senseful amount. More tapers will have no great influence on the final spectrum but increase the
     calculation time. Use fewer tapers for a faster calculation.
    :param time_bandwidth_div: Smoothing amount, should be between 1 and Nsamps/2.
    :param sine: if True, will use sine_psd instead of multitaper, sine method should be used for sharp cutoffs or
     deep valleys, or small sample sizes
    :param recsec: if True, will plot spectra each in their own plot, otherwise all spectra will be in the same plot
    :param logx: if True, will use log scale for x axis
    :param logy: if True, will use log scale for y axis
    :param xunits: if 'Hz' x axis will be frequency, if 'sec' x axis will be period
    :param xlim: tuple of xlims like (0., 100.)
    :param yunits: string defining y units - units should be input time series units squared per Hz
    :param groupstations: only if recsec is True - if True, will plot all channels from same station on one plot
    """

    st = Stream(st)  # in case it's a trace
    if detrend is not None:
        st.detrend(detrend)

    if recsec:
        stas = unique_list([trace.stats.station for trace in st])
        lent = len(stas)
        fig, axes = plt.subplots(lent, sharex=True, sharey=False, figsize=(10, min(10, 2*(lent+1))))
        if lent == 1:
            axes = [axes]
    else:
        fig, ax = plt.subplots(1, figsize=(8, 5))

    if logx:
        logx = 'log'
    else:
        logx = 'linear'
    if logy:
        logy = 'log'
    else:
        logy = 'linear'

    freqs, amps = sigproc.multitaper(st, number_of_tapers=number_of_tapers, time_bandwidth=time_bandwidth, sine=sine)
    i = 0
    for amp, freq in zip(amps, freqs):
        if colors1 is not None:
            color = colors1[i]
        else:
            color = np.random.rand(3,)
        if recsec:
            ind = stas.index(st[i].stats.station)
            ax = axes[ind]
            if st[i].stats.channel[-1] == 'Z':
                linestyle = '-'
            elif st[i].stats.channel[-1] == 'N' or st[i].stats.channel[-1] == '1':
                linestyle = '--'
            else:
                linestyle = ':'
            if xunits == 'Hz':
                ax.plot(freq, amp, label=st[i].id, color=color, linestyle=linestyle)
                xun = 'Frequency (Hz)'
            elif xunits == 'sec':
                ax.plot(1./freq, amp, label=st[i].id, color=color, linestyle=linestyle)
                xun = 'Period (sec)'
            if i == lent - 1:
                ax.set_xlabel(xun, fontsize=labelsize)
            ax.legend(fontsize=labelsize - 2.)
        else:
            if yunits is not None:
                ax.set_ylabel('Power spectral density (%s)' % yunits)
            else:
                ax.set_ylabel('Power spectral density')

            if i == len(st)-1:
                plt.legend(fontsize=labelsize)
                ax.set_xlabel(xun, fontsize=labelsize)
            if xunits == 'Hz':
                ax.plot(freq, amp, label=st[i].id, color=color)
                xun = 'Frequency (Hz)'
            elif xunits == 'sec':
                ax.plot(1./freq, amp, label=st[i].id, color=color)
                xun = 'Period (sec)'
        ax.set_yscale(logy)
        ax.set_xscale(logx)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=labelsize)
        max_yticks = 4
        yloc = plt.LogLocator(numticks=max_yticks)
        ax.yaxis.set_major_locator(yloc)
        if i == 0:
            mcs = ('%.2f' % (st[0].stats.starttime.microsecond/10.**6)).replace('0.', '')
            timeprint1 = st[0].stats.starttime.strftime('%Y-%m-%dT%H:%M:%S.') + mcs
            mcs = ('%.2f' % (st[0].stats.endtime.microsecond/10.**6)).replace('0.', '')
            timeprint2 = st[0].stats.endtime.strftime('%Y-%m-%dT%H:%M:%S.') + mcs
            ax.set_title('Time period: %s to %s' % (timeprint1, timeprint2), fontsize=labelsize)
        i += 1

    if recsec:
        plt.subplots_adjust(hspace=0.)  # reduce vertical space between plots
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)  # Turn of x labels where not needed

        if yunits is not None:
            fig.text(0.04, 0.5, 'Power spectral density (%s)' % yunits, va='center', rotation=90)
        else:
            fig.text(0.04, 0.5, 'Power spectral density', va='center', rotation=90)

    plt.xlabel(xun)
    plt.subplots_adjust(bottom=0.15)

    if render:
        plt.show(fig)
    else:
        plt.close(fig)

    return freqs, amps, fig


def make_spectrogram(st, detrend=mlab.detrend_linear, indfirst=0, maxtraces=10, wlen=None, norm=True,
                     overperc=0.85, log1=True, maxPower=None, minPower=None, freqmax=None,
                     colorb=False, labelsize=12, figsize=None, render=True):
    """
    Plot spectrogram (opens new figure)
    USAGE
    fig, axes = make_spectrogram(st, detrend=mlab.detrend_linear, indfirst=0, maxtraces=10, wlen=None, norm=True,
                     overperc=0.85, log1=True, maxPower=None, minPower=None, freqmax=None,
                     colorb=False, labelsize=12, figsize=None, render=True)
    INPUTS
    st = obspy stream containing data to plot
    detrend = type of detrending to use from matplotlib mlab commands (mlab.detrend_linear, mlab.detrend_mean, or mlab.dtrend_none)
    indfirst = index of first trace to plot
    maxtraces = maximum number of traces to plot
    wlen = window length (sec) for spectrogram generation
    overperc = overlap percentage (1.0 is total overlap)
    log1 = True for log scaling of amplitude, False for linear
    maxPower = upper limit at which to saturate color scale
    minPower = lower limit at which to saturate color scale
    freqmax = maximum frequency to show along y-axis
    colorb = True, show colorbar, False, don't.

    OUTPUTS
    fig = figure handle
    axes = axes handles

    TO DO
    if view window is longer than a certain number of seconds, change to minutes or hours
    """
    if figsize is None:
        figsize = (12, min(18, 2*maxtraces))

    maxtraces = min(len(st), maxtraces)
    fig, axes = plt.subplots(maxtraces, sharex=True, sharey=False, figsize=figsize)

    if maxtraces == 1:
       axes = [axes]

    mcs = ('%.2f' % (st[0].stats.starttime.microsecond/10.**6)).replace('0.', '')
    timeprint = st[0].stats.starttime.strftime('%Y-%m-%dT%H:%M:%S.') + mcs
    title1 = 'Start time: %s (UTC)' % timeprint
    if norm:
        title1 = title1 + ' - normalized to maxes'
    else:
        title1 = title1 + ' - same amplitude scaling'

    for i, st1 in enumerate(st[indfirst:indfirst+maxtraces]):
        if wlen is None:
            wlen = np.min([(st1.stats.endtime - st1.stats.starttime)/50., st1.stats.sampling_rate/25.])
        NFFT = int(nextpow2(wlen*st1.stats.sampling_rate))
        noverlap = int(overperc*NFFT)
        Pxx, freq, time = mlab.specgram(st1.data, NFFT=NFFT, Fs=st1.stats.sampling_rate,
                                        detrend=detrend, noverlap=noverlap)
        Pxx = np.flipud(Pxx)
        # center bin
        # calculate half bin width
        halfbin_time = (time[1] - time[0]) / 2.0
        halfbin_freq = (freq[1] - freq[0]) / 2.0
        extent = (time[0] - halfbin_time, time[-1] + halfbin_time,
                  freq[0] - halfbin_freq, freq[-1] + halfbin_freq)

        minP = None
        maxP = None

        if norm or maxPower is not None:
            if maxPower is None:
                maxP = np.abs(st1.max())**2.
                vmax = maxP
            else:
                maxP = maxPower
                vmax = None
            if minPower is None:
                minP = maxP/1e6
                vmin = minP
            else:
                minP = minPower
                vmin = None
        else:
            maxP = np.max(np.abs(st.max()))**2.
            minP = maxP/1e6
            vmin = minP
            vmax = maxP

        print(('Max power: %1.2e, Min power: %1.2e\n' % (maxP, minP)))

        if log1 is True and minP is not None and maxP is not None:
            vmin = np.log10(minP)
            vmax = np.log10(maxP)

        ylims = (0, st1.stats.sampling_rate/2.)
        if log1 is True:
            im = axes[i].imshow(np.log10(Pxx), interpolation="nearest", extent=extent,
                                vmin=vmin, vmax=vmax)
        else:
            im = axes[i].imshow(Pxx, interpolation="nearest", extent=extent,
                                vmin=vmin, vmax=vmax)
        if i == 0:
            axes[i].set_title(title1, fontsize=labelsize + 1)
        axes[i].axis('tight')
        axes[i].grid(True)
        axes[i].tick_params(labelright=True, labelsize=labelsize)
        if freqmax is None:
            axes[i].set_ylim(ylims)
        else:
            axes[i].set_ylim([0, freqmax])
        axes[i].set_ylabel(st1.stats.station+'.'+st1.stats.channel+'.'+st1.stats.location+'.'+st1.stats.network, fontsize=labelsize, rotation=0)
        axes[i].yaxis.set_label_coords(-0.15, 0.5)
    plt.subplots_adjust(hspace=0.1, left=0.23)  # reduce vertical space between plots
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    axes[-1].tick_params(axis='both', which='major')
    plt.xlabel('Time (sec)', fontsize=labelsize)
    if colorb is True and maxPower is not None and minPower is not None:
        plt.subplots_adjust(right=0.8)  # make room for colorbar
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax)

    if render:
        plt.show(fig)
    else:
        plt.close(fig)
    #plt.tight_layout()
    return fig, axes


class InteractivePlot:
    """
    Class for interactive plotting using recsec
    USAGE:
    zp = reviewData.InteractivePlot(st, fig=None, indfirst=0, maxtraces=10, norm=True, xlim=None, ylim=None, scalfact=1.,
                 cosfilt=(0.01, 0.02, 20, 30), water_level=60, output='VEL', textline=['>', '>', '>', '>', '>'],
                 menu=None, quickdraw=True, processing=None, taper=0.05):

    INPUTS
    see recsec for recsec inputs
    fig = figure handle output by recsec
    st = obspy stream to interact with
    indfirst = index of first trace to show
    maxtraces = maximum number of traces to show
    cosfilt = cosine filter used for station corrections, tuple of four floats defining corners of cosine in Hz, e.g. (0.01, 0.02, 20., 30.), set to None for no pre-filtering
    water_level = water level, in dB, for station corrections
    output = units of output, 'VEL', 'DISP' or 'ACC'
    textline = lines of text that will show when plot is first opened
    menu = if None, will show default menu, otherwise will show menu
    quickdraw = if True, will downsample strategically to speed up plotting
    processing = if None, won't show the processing that has been applied to the data, otherwise will show whatever is entered for processing
    taper = taper to apply before filtering, if None, no tapering will be applied (default 0.05, 5% taper)
    specg = dictionary of inputs for make_spectrogram
    vlines = locations of vertical lines

    OUTPUTS - everything that is stored in self (called zp in above example) can be accessed after exiting, here are a few of the most useful ones
    zp.st_current = processed data in time window that was used
    zp.deleted = traces that were deleted during processing
    zp.picks = dictionary of time picks
    zp.st_original = original data

    TODO
    Absolute timeline option
    Add vertical lines
    Make picking phases a width of uncertainty rather than weight
    Use RectangleSelector widget to zoom
    Backspace functionality for inputs
    Interactively editable spectrogram options
    Fix Box zooming
    Make axis change from seconds to minutes or hours as needed

            make_spectrogram(st_temp, detrend=mlab.detrend_linear, maxtraces=len(st_temp),
                             wlen=((st_temp[0].stats.endtime-st_temp[0].stats.starttime)/150.),
                             overperc=0.9, log1=True, maxPower=100000, minPower=100,
                             freqmax=st_temp[0].stats.sampling_rate/2., colorb=True)

    """

    def __init__(self, st, fig=None, indfirst=0, maxtraces=10, norm=True,
                 xlim=None, ylim=None, scalfact=1., cosfilt=(0.01, 0.02, 20, 30),
                 water_level=60, output='VEL', textline=['>', '>', '>', '>', '>'],
                 menu=None, quickdraw=True, processing=None, taper=None,
                 tscale='relative', vlines=None, picktimes=None,
                 specg = {'detrend': mlab.detrend_linear, 'wlen': None, 'norm': True, 'overperc': 0.85, 'log1': True,
                          'maxPower': None, 'minPower': None, 'freqmax': None}):
        """
        Initializes the class with starting values
        """
        try:
            self.reset  # reset the class each time it's initialized
        except:
            pass  # New instance

        self.ind = 0  # index for current zoom position
        self.zflag = 0  # index used for zooming
        self.azflag = 0  # index used for box zoomming
        self.aflag = 0  # index used for amplitude selection
        self.lflag = 0  # used for pseudoenergy selection
        self.numflag = None  # flag for keeping track of numbers
        self.number = ''
        self.input1 = 0
        self.input2 = 0
        self.cosfilt = cosfilt
        self.water_level = water_level
        self.output = output
        self.deleted = []  # keeps track of traces that were deleted
        self.normflag = norm
        self.scalfact = scalfact
        self.picktime = 0
        self.phasep = None
        self.pweight = 0
        self.picksta = None
        self.picknumber = 0
        self.picks = {}
        self.init = 0
        self.st_original = st.copy()
        self.st = st.copy()
        self.st_current = st.copy()
        self.st_last = st.copy()
        self.tempdelsta = None  # hold station to delete until it's confirmed
        self.indfirst = indfirst
        self.maxtraces = maxtraces
        self.tmin = min([trace.stats.starttime for trace in st])
        self.tmax = max([trace.stats.starttime for trace in st])
        self.print1 = textline
        self.menu_print = menu
        self.processing = processing
        self.processing_print = processing
        self.env = False  # whether plot is an envelope or not
        self.quickdraw = quickdraw
        self.taper = taper
        self.tscale = tscale
        self.specg = specg
        self.vlines = vlines
        if picktimes is not None:
            self.picktimes = list(picktimes)
            self.picktimes_original = list(picktimes.copy())
        else:
            self.picktimes = None
            self.picktimes_original = None
        if taper is not None:
            if 60/(self.tmax-self.tmin) > 0.05:
                self.taper = 60./(self.tmax-self.tmin)  # Taper on first minute if the signal length is really long
            self.dotaper = True
        else:
            self.dotaper = False
        self.menu = """
        up - double scaling
        down - half scaling
        right - move forward
        left - go back
        A - make amplitude pick
        B - box zoom
        C - station correction
        D - scroll down a trace
        E - envelopes
        F - bandpass filter
        G - spectrogram current window
        H - show or hide help menu (toggle)
        I - show or hide processing summary (toggle)
        J - change number of traces shown
        K - kurtosis (not working yet)
        L - make pseudoenergy pick
        M - change norm mode
        O - delete selected trace (X to reset)
        P - make P phase pick (N cancels)
        Q - quit
        R - previous view
        S - make S phase pick (N cancels)
        T - print timestamp
        U - scroll up a trace
        V - undo data change
        W - change window len
        X - reset to original data
        Z - zoom x-only
        + - page down
        - - page up
        @ - toggle quickdraw (default on, may be slow if you turn it off)
        ! - save figure (without textbox) using same dimensions of current figure
        $ - make spectra
        # - show or hide scale bars
        """
        if fig is None:
            self.fig = recsec(self.st_current, xlim=xlim,
                              ylim=ylim, scalfact=self.scalfact,
                              update=False, fighandle=None, tscale=self.tscale,
                              norm=self.normflag, indfirst=self.indfirst,
                              maxtraces=self.maxtraces, textline=self.print1,
                              menu=self.menu_print, processing=self.processing_print,
                              textbox=True, quickdraw=self.quickdraw,
                              vlines=self.vlines, picktimes=self.picktimes)
        else:
            self.fig = fig
        self.axbox = self.fig.get_axes()[0]
        self.ax = self.fig.get_axes()[1]
        self.xlims = list(self.ax.get_xlim())  # saves history or x, y coords
        self.ylims = list(self.ax.get_ylim())
        self.connect()

    def closed(self, evt):
        """
        disconnect and continue code
        """
        self.fig.canvas.mpl_disconnect(self.cidkey)
        self.fig.canvas.mpl_disconnect(self.cidscroll)
        self.fig.canvas.stop_event_loop()
        print('Exiting interactive plotting')
        self.print1 = ['>', '>', '>', '>', '>']

    def connect(self):
        """
        connect to needed events, suspends code until figure is closed
        """
        self.cidkey = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.cidscroll = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)  # turn off the automatic keys
        print((self.menu))
        self.fig.canvas.mpl_connect('close_event', self.closed)
        self.fig.canvas.start_event_loop(timeout=6000)  # -1
        try:
            self.fig.canvas.manager.window.raise_()
        except:
            print('Using %s backend, interactive plotting may not work. Must use Qt backend' % matplotlib.get_backend())

    def disconnect(self):
        """
        disconnect and continue code
        """
        self.fig.canvas.mpl_disconnect(self.cidkey)
        self.fig.canvas.mpl_disconnect(self.cidscroll)
        self.fig.canvas.stop_event_loop()
        self.print1 = ['>', '>', '>', '>', '>']

    def on_key(self, event):
        """
        Event triggers from pressing keys on keyboard
        """
        plt.sca(self.ax)
        redraw = False
        update = True
        #xlims = np.sort(self.xlims[-2:])
        ylims = None

        if event.xdata is None:
            temp = 'The recsec plot may not be active, click on figure and move it slightly to make it active and try again'
            event.key = '.'
            self.phasep = None
            self.numflag = None
            print(temp)
            self.print1.append('> '+temp)
            redraw = True

        # Sorting out entering info
        if event.key == 'enter':
            if self.numflag is None:
                pass
            if self.numflag == 'F1':  # got two inputs so now apply bandpass filter
                try:
                    self.input2 = float(self.number)
                    print(('Filtering current data between %1.2f and %1.2f Hz' % (self.input1, self.input2)))
                    self.print1.append('> Filtering current data between %1.2f and %1.2f Hz' % (self.input1, self.input2))
                    self.number = ''
                    self.st_current.detrend('demean')
                    if self.taper is not None:
                        self.st_current.taper(max_percentage=self.taper, type='cosine')
                    self.st_current.filter('bandpass', freqmin=self.input1, freqmax=self.input2,
                                           corners=2, zerophase=False)
                    redraw = True
                    self.numflag = None
                except:
                    temp = 'Failed, resetting'
                    self.number = ''
                    self.numflag = None
                    print(temp)
                    self.print1.append('> '+temp)
            elif self.numflag == 'F':  # need one more input for filter
                try:
                    self.input1 = float(self.number)
                    self.number = ''
                    self.numflag = 'F1'
                    temp = ('Type upper freq limit for filter and hit enter')
                    print(temp)
                    self.print1.append('> '+temp)
                except:
                    temp = 'Failed, resetting'
                    self.number = ''
                    self.numflag = None
                    print(temp)
                    self.print1.append('> '+temp)
            elif self.numflag == 'P':
                try:
                    self.input1 = float(self.number)
                    self.number = ''
                    self.pweight = self.input1
                    temp = ('Pick of %s phase at %s at %f sec weighted as %i, hit Y to keep it' % (self.phasep, self.picksta, self.picktime, self.pweight))
                    print(temp)
                    self.print1.append('> '+temp)
                    self.numflag = 'PW'
                except:
                    temp = 'Failed, resetting1'
                    self.number = ''
                    self.numflag = None
                    print(temp)
                    self.print1.append('> '+temp)
            elif self.numflag == 'W':
                try:
                    self.input1 = float(self.number)
                    self.number = ''
                    self.numflag = None
                    xlim = self.ax.get_xlim()
                    #save to lim history
                    self.xlims.append(xlim[0])
                    self.xlims.append(xlim[0]+self.input1)
                    self.ylims.append(self.ax.get_ylim()[0])
                    self.ylims.append(self.ax.get_ylim()[1])
                    redraw = True
                except:
                    temp = 'Failed, resetting'
                    self.number = ''
                    self.numflag = None
                    print(temp)
                    self.print1.append('> '+temp)
            elif self.numflag == 'J':
                try:
                    self.maxtraces = int(self.number)
                except:
                    print('failed to change maxtraces, resetting')
                self.number = ''
                self.numflag = None
                redraw = True
            elif self.numflag == '!':
                print('creating figure')
                figsize = self.fig.get_size_inches()
                textpos = self.axbox.get_position()
                figsize = tuple([figsize[0], figsize[1] - textpos.height*figsize[1]])
                figprint = recsec(self.st_current, xlim=np.sort(self.xlims[-2:]),
                                  ylim=ylims, scalfact=self.scalfact,
                                  update=False, fighandle=None,
                                  norm=self.normflag, indfirst=self.indfirst,
                                  maxtraces=self.maxtraces, tscale=self.tscale,
                                  menu=None, processing=None,
                                  quickdraw=False, textbox=False,
                                  figsize=figsize, vlines=self.vlines, picktimes=self.picktimes)
                figprint.savefig(self.number+'.png', format='png')
                plt.close(figprint)
                print(('figure %s saved' % self.number))
                self.numflag = None
                self.number = ''

        #keep track of numbers that are typed in an print them as they are typed in
        if self.numflag is not None:
            if self.numflag in '!,F,J,P,W,F1' and event.key != 'enter':
                if event.key == 'backspace':
                    if self.number != '':
                        self.number = self.number[:-1]
                else:
                    self.number = self.number + event.key
                event.key = '.'
                #try:
                #    int(event.key)
                #    self.number = self.number + event.key
                #except:
                #    self.number = self.number + event.key
                #if event.key == '.':
                #    self.number = self.number + event.key
                if len(self.number) > 1:
                    self.print1[-1] = '> '+self.number
                else:
                    self.print1.append('> '+self.number)

        #These lines fix things if you don't have a second required input for something like zoom or the figure isn't active etc.
        if event.key.upper() != 'Z':
            if self.zflag == 1:
                self.xlims.pop()
                self.ylims.pop()
            self.zflag = 0
        if event.key.upper() != 'B':
            if self.azflag == 1:
                self.xlims.pop()
                self.ylims.pop()
            self.azflag = 0
        if event.key.upper() != 'A' and self.aflag == 1:
            #if don't get second A for amplitude pick, reset
            self.phasep = None
            self.numflag = None
        if event.key.upper() != 'L' and self.lflag == 1:
            #if don't get second L for pseudoenergy pick, reset
            self.phasep = None
            self.numflag = None

        #These are for scrolling
        if event.key == 'up':  # expand scale by factor of 2
            self.scalfact = self.scalfact*2
            redraw = True

        if event.key == 'down':  # shrink scale by factor of 2
            self.scalfact = self.scalfact*0.5
            redraw = True

        if event.key == 'right':  # advance to next window (same as scroll up)
            cur_xlim = self.ax.get_xlim()
            xwid = cur_xlim[1]-cur_xlim[0]
            #save to lim history
            self.xlims.append(cur_xlim[1])
            self.xlims.append(cur_xlim[1]+xwid)
            self.ylims.append(self.ax.get_ylim()[0])
            self.ylims.append(self.ax.get_ylim()[1])
            self.ind += 2
            redraw = True

        if event.key == 'left':  # go back to previous window (same as scroll down)
            cur_xlim = self.ax.get_xlim()
            xwid = cur_xlim[1]-cur_xlim[0]
            #save to lim history
            self.xlims.append(cur_xlim[0]-xwid)
            self.xlims.append(cur_xlim[0])
            self.ylims.append(self.ax.get_ylim()[0])
            self.ylims.append(self.ax.get_ylim()[1])
            self.ind += 2
            redraw = True

        #Regular menu items
        if event.key.upper() == 'A':  # make amplitude pick
            if self.phasep is None:
                self.phasep = event.key.upper()
                self.picktime = event.xdata
                self.aflag = 1
                temp = ('hit A at next (later) part of signal of which you want absmax')
                print(temp)
                self.print1.append('> '+temp)
            elif self.phasep == 'A':
                self.aflag = 0
                #find which station click was closest to
                keyvals = list(self.fig.stationsy.keys())
                idx = np.absolute(event.ydata-np.array(keyvals)).argmin()
                self.picksta = self.fig.stationsy[keyvals[idx]]
                if event.xdata > self.picktime:
                    self.picktime = [self.picktime, event.xdata]
                    temp = self.picksta.split('.')
                    sta = temp[0]
                    chan = temp[1]
                    loc = temp[2]
                    net = temp[3][:2]
                    tempst = [trace for trace in self.st_current if trace.stats.station == sta and trace.stats.channel == chan and trace.stats.location == loc and trace.stats.network == net]
                    tempst = tempst[0]
                    #find amplitude at that time at that station and save it under weight placeholder
                    tdiff = tempst.stats.starttime - self.tmin
                    tvec = np.linspace(0, (len(tempst.data)-1)*1/tempst.stats.sampling_rate, num=len(tempst.data)) + tdiff
                    self.pweight = np.amax(np.absolute([val for i, val in enumerate(tempst.data) if tvec[i] > min(self.picktime) and tvec[i] < max(self.picktime)]))
                    temp = ('Amplitude pick saved at %s at %1.2f to %1.2f sec with amplitude of %1.1E'
                            % (self.picksta, self.picktime[0], self.picktime[1], self.pweight))
                    print(temp)
                    self.print1.append('> '+temp)
                    self.picks[self.picknumber] = {'stachan': self.picksta,
                                                   'picktime': [self.tmin + self.picktime[0],
                                                                self.tmin + self.picktime[1]],
                                                   'phase': self.phasep, 'weight': self.pweight}
                    self.picknumber += 1
                else:
                    temp = 'Amplitude pick starttime later than endtime - not saving pick'
                    print(temp)
                    self.print1.append('> '+temp)
                self.phasep = None
                self.ptime = 0
                self.pweight = 0
                self.picktime = 0

        if event.key.upper() == 'B':
            #do xy zoom box
            if self.azflag == 0:
                self.xlims.append(event.xdata)
                self.ylims.append(event.ydata)
                self.azflag = 1
                temp = ('hit b at the next corner of zoom window')
                print(temp)
                self.print1.append('> '+temp)

            elif self.azflag == 1:  # once you get a second b, replot window
                self.xlims.append(event.xdata)
                self.ylims.append(event.ydata)
                self.ind += 2
                self.azflag = 0
                ylims = np.sort(self.ylims[-2:])
                redraw = True
                #if self.quickdraw:
                #    redraw = True
                #else:
                #    self.ax.set_xlim(np.sort(self.xlims[-2:]))
                #    self.ax.set_ylim(ylims)
                #    self.ax.figure.canvas.draw()

        if event.key.upper() == 'C':  # do station correction and replot
            try:
                self.st_current = self.st.copy()
                try:
                    self.st_current.remove_response(output=self.output, pre_filt=self.cosfilt,
                                                    water_level=self.water_level, taper=self.dotaper, taper_fraction=self.taper)
                except:
                    print('Failed to do bulk station correction, trying one at a time')
                    self.st_current = self.st.copy()  # Start with fresh data
                    removeid = []
                    for trace in self.st_current:
                        try:
                            trace.remove_response(output=self.output, pre_filt=self.cosfilt,
                                                  water_level=self.water_level, taper=self.dotaper, taper_fraction=self.taper)
                        except:
                            print(('Failed to remove response for %s, deleting this station' % (trace.stats.station + trace.stats.channel,)))
                            removeid.append(trace.id)
                    for rmid in removeid:  # Delete uncorrected ones
                        for tr in self.st_current.select(id=rmid):
                            self.st_current.remove(tr)
                redraw = True
                if self.cosfilt is None:
                    temp = ('Removed station response with just water_level of %1.0f' % (self.water_level,))
                else:
                    temp = ('Removed station response to %s with cosine filter %1.3f %1.3f %1.2f %1.2f'
                            % (self.output, self.cosfilt[0], self.cosfilt[1], self.cosfilt[2], self.cosfilt[3]))
                print(temp)
                self.print1.append('> '+temp)
                #print self.st_current[0].data.max()

            except Exception as e:
                temp = e
                print(temp)
                #self.print1.append('> '+temp)
                temp = ('failed to do station correction')
                print(temp)
                self.print1.append('> '+temp)

        if event.key.upper() == 'D' and len(self.st) > self.maxtraces:  # scroll down a trace
            self.indfirst = min(self.indfirst+1, len(self.st_current)-self.maxtraces)
            redraw = True

        if event.key.upper() == 'E':  # make envelopes of whatever is current data
            for i, _ in enumerate(self.st):
                self.st_current[i].data = filte.envelope(self.st_current[i].data)
            redraw = True
            self.env = True
            temp = ('Showing envelopes, press X to reset')
            print(temp)
            self.print1.append('> '+temp)

        if event.key.upper() == 'F':  # bandpassfilter and replot
            self.numflag = 'F'
            temp = ('Type lower freq limit for filter and hit enter')
            print(temp)
            self.print1.append('> '+temp)

        if event.key.upper() == 'G':  # make spectrogram of current window - open in new fig
            print('Generating spectrogram of current window')
            xlimtemp = self.ax.get_xlim()
            st_temp = self.st.copy()
            st_temp = st_temp.slice(self.tmin + xlimtemp[0], self.tmin + xlimtemp[1])
            st_temp = st_temp[self.indfirst:self.indfirst+self.maxtraces+1]
            if self.specg['detrend'] is None:
                self.specg['detrend'] = mlab.detrend_linear
            if self.specg['freqmax'] is None:
                self.specg['freqmax'] = st_temp[0].stats.sampling_rate/2.
#            if self.specg['minpower'] is None:
#                self.specg['minpower'] = 10
#            if self.specg['maxpower'] is None:
#                self.specg['maxpower'] = 1000000
#            if self.specg['wlen'] is None:
                self.specg['wlen'] = ((st_temp[0].stats.endtime-st_temp[0].stats.starttime)/150.)

            make_spectrogram(st_temp, colorb=True, maxtraces=len(st_temp), **self.specg)

        if event.key.upper() == 'H':  # toggle help menu
            if self.menu_print is not None:
                self.menu_print = None
            else:
                self.menu_print = self.menu
            print((self.menu))
            redraw = True
            #update = False

        if event.key.upper() == 'I':  # toggle processing summary
            if self.processing_print is not None:
                self.processing_print = None
            else:
                self.processing_print = self.processing
            redraw = True
            #update = False

        if event.key.upper() == 'J':  # change # of maxtraces
            self.numflag = 'J'
            temp = ('enter new number of traces to show')
            print(temp)
            self.print1.append('> '+temp)

        if event.key.upper() == 'K':  # plot continuous kurtosis
            temp = ('kurtosis not implemented yet')
            print(temp)
            self.print1.append('> '+temp)
            #for i, junk in enumerate(self.st):
            #    temp = opsig.kurtosis(self.st_current[i])
            #    self.st_current[i].data = temp
            #self.fig = update_recsec(self.st_current,self.fig,scalfact=self.scalfact)
            #self.ax = fig.gca()
            #self.ax.set_xlim(np.sort(self.xlims[-2:]))
            #print 'Opening new window showing current data recursive kurtosis'
            #self.ax.figure.canvas.draw()

        if event.key.upper() == 'L':  # make pseduoenergy pick (velocity squared)
            #self.cursor = Cursor(self.ax,useblit=True,color='red',linewidth=2)
            if self.phasep is None:
                self.phasep = event.key.upper()
                self.picktime = event.xdata
                self.lflag = 1
                temp = ('hit L at next (later) part of signal of which you want psuedoenergy calc')
                print(temp)
                self.print1.append('> '+temp)

            elif self.phasep == 'L':
                self.lflag = 0
                #find which station click was closest to
                keyvals = list(self.fig.stationsy.keys())
                idx = np.absolute(event.ydata-np.array(keyvals)).argmin()
                self.picksta = self.fig.stationsy[keyvals[idx]]
                self.picktime = [self.picktime, event.xdata]
                temp = self.picksta.split('.')
                sta = temp[0]
                chan = temp[1]
                loc = temp[2]
                net = temp[3][:2]
                tempst = [trace for trace in self.st_current if trace.stats.station == sta and trace.stats.channel == chan and trace.stats.location == loc and trace.stats.network == net]
                tempst = tempst[0].slice(self.tmin+min(self.picktime), self.tmin+max(self.picktime))
                dat = tempst.data
                samprate = tempst.stats.sampling_rate
                tvec1 = np.linspace(0, (len(dat)-1)*1/samprate, num=len(dat))
                dat = filte.envelope(dat)
                self.pweight = np.trapz(dat**2, x=tvec1)  # velocity squared
                temp = ('Pseudoenergy pick saved at %s at %1.2f to %1.2f sec with energy of %1.1E'
                        % (self.picksta, self.picktime[0], self.picktime[1], self.pweight))
                print(temp)
                self.print1.append('> '+temp)
                self.picks[self.picknumber] = {'stachan': self.picksta,
                                               'picktime': [self.tmin+self.picktime[0],
                                                            self.tmin+self.picktime[1]],
                                               'phase': self.phasep, 'weight': self.pweight}
                self.picknumber += 1
                self.phasep = None
                self.ptime = 0
                self.pweight = 0
                #self.cursor = None

        if event.key.upper() == 'M':  # change normalization mode
            self.scalfact = 1
            if self.normflag is True:
                self.normflag = False
            elif self.normflag is False:
                self.normflag = True
            redraw = True

        if event.key.upper() == 'O':  # delete trace and keep track of which ones were deleted and output it
            self.numflag = 'O'
            #find which station click was closest to
            keyvals = list(self.fig.stationsy.keys())
            idx = np.absolute(event.ydata-np.array(keyvals)).argmin()
            #staname = st1.stats.station+'.'+st1.stats.channel+'.'+st1.stats.network
            self.tempdelsta = self.fig.stationsy[keyvals[idx]]
            temp = ('Delete trace of %s ? Hit Y to confirm' % (self.tempdelsta,))
            print(temp)
            self.print1.append('> '+temp)

        if event.key.upper() == 'P' or event.key.upper() == 'S':
            self.phasep = event.key.upper()
            self.numflag = 'P'
            self.picktime = event.xdata
            #find which station click was closest to
            keyvals = list(self.fig.stationsy.keys())
            idx = np.absolute(event.ydata-np.array(keyvals)).argmin()
            self.picksta = self.fig.stationsy[keyvals[idx]]
            temp = ('assign weight and hit enter')
            print(temp)
            self.print1.append('> '+temp)

        if event.key.upper() == 'Q':  # quit interactive plot and output st as processed
            temp = ('Closing figure, outputting data in current view to class')
            print(temp)
            self.print1.append('> '+temp)
            xlims = self.ax.get_xlim()
            plt.close(self.fig)
            self.quit = True
            self.st_current = self.st_current.slice(self.tmin + xlims[0],
                                                    self.tmin + xlims[1])
            self.disconnect()

        if event.key.upper() == 'R':  # go back to previous zoom
            if self.ind > 0:
                self.xlims.pop()  # pop off last two values from the history
                self.xlims.pop()
                self.ylims.pop()
                self.ylims.pop()
                self.ind += -2  # change index back
                redraw = True
            else:
                temp = 'already at first zoom window'
                print(temp)
                self.print1.append('> '+temp)

        if event.key.upper() == 'T':  # print timestamp
            try:
                temp = str(self.tmin + event.xdata)
                print(temp)
                self.print1.append('> '+temp)
            except:
                print('cannot print timestamp - you may not be in the figure')

        if event.key.upper() == 'U' and self.maxtraces < len(self.st):  # scroll up a trace
            self.indfirst = max(self.indfirst-1, 0)
            redraw = True

        if event.key.upper() == 'V':  # undo last change to data
            self.st_current = self.st_last
            temp = 'going back to last data plotted'
            print(temp)
            self.print1.append('> '+temp)
            redraw = True

        if event.key.upper() == 'W':  # change window width
            self.numflag = 'W'
            temp = 'enter new window length in seconds'
            print(temp)
            self.print1.append('> '+temp)

        if event.key.upper() == 'X':  # reset current data to original data
            self.st_current = self.st_original.copy()
            if self.picktimes is not None:
                self.picktimes = self.picktimes_original.copy()
            temp = 'resetting to original data'
            print(temp)
            self.print1.append('> '+temp)
            redraw = True
            #update = False
            self.deleted = []  # resetting data also resets those that were deleted

        if event.key.upper() == 'Y' and self.numflag == 'PW':
            self.numflag = None
            #save the pick in adictionary, use UTCDateTime to add seconds to start time of st
            self.picks[self.picknumber] = {'stachan': self.picksta, 'picktime': self.st[0].stats.starttime + self.picktime, 'phase': self.phasep, 'weight': self.pweight}
            self.picknumber += 1
            self.phasep = None
            self.ptime = 0
            self.pweight = 0
            temp = 'pick saved'
            print(temp)
            self.print1.append('> '+temp)
        elif event.key.upper() == 'Y' and self.numflag == 'O':
            self.numflag = None
            temp = self.tempdelsta.split('.')
            for i, st1 in enumerate(self.st_current):
                if st1.stats.station == temp[0] and st1.stats.channel == temp[1] and st1.stats.location == temp[2] and st1.stats.network == temp[3].split('-')[0].strip():
                    self.deleted.append(self.tempdelsta)
                    self.tempdelsta = None
                    self.st_current.pop(i)
                    self.st.pop(i)
                    if self.picktimes is not None:
                        self.picktimes.pop(i)
                    temp = ('%s deleted, press x to reset data' % (self.deleted[-1],))
                    print(temp)
                    self.print1.append('> '+temp)
                    redraw = True
                    #update = False

        if event.key.upper() == 'N' and self.numflag == 'PW':
            temp = 'deleting pick'
            print(temp)
            self.print1.append('> '+temp)
            self.phasep = None
            self.ptime = 0
            self.pweight = 0
            self.numflag = None

        if event.key.upper() == 'Z':
            #save current zoom before changing
            if self.zflag == 0:
                self.xlims.append(event.xdata)
                self.ylims.append(self.ax.get_ylim())
                self.zflag = 1
                temp = 'hit z at the next corner of zoom window'
                print(temp)
                self.print1.append('> '+temp)
            elif self.zflag == 1:  # once you get a second Z, replot window
                self.xlims.append(event.xdata)
                self.ylims.append(self.ax.get_ylim())
                self.ind += 2
                self.zflag = 0
                if self.quickdraw:
                    redraw = True
                else:
                    self.ax.set_xlim(np.sort(self.xlims[-2:]))
                    self.ax.figure.canvas.draw()

        self.axbox.clear()
        self.axbox.set_yticks([])
        self.axbox.set_xticks([])
        self.axbox.text(0.01, 0.99, '\n'.join(self.print1[-5:]),
                        transform=self.axbox.transAxes, fontsize=12,
                        verticalalignment='top')
        self.axbox.figure.canvas.draw()

        if event.key.upper() == '@':
            self.quickdraw = not self.quickdraw
            if self.quickdraw is False:
                self.print1.append('> quickdraw off')
            else:
                self.print1.append('> quickdraw on')
            redraw = True

        if event.key.upper() == '!':
            #print figure
            self.numflag = '!'
            temp = ('enter filename to save figure - be sure to type in figure, not terminal, and leave off extension')
            print(temp)
            self.print1.append('> '+temp)
            redraw = True

        if event.key.upper() == '+':  # page down
            self.indfirst = min(self.indfirst+self.maxtraces, len(self.st_current)-self.maxtraces)
            redraw = True

        if event.key.upper() == '-':  # page up
            self.indfirst = max(self.indfirst-self.maxtraces, 0)
            redraw = True

        self.st_last = self.st_current.copy()

        if redraw is True:
            #self.ax.clear()
            if self.picktimes is not None:
                # delete the previous lines
                templi = self.ax.lines
                indx5 = []
                for i, li in enumerate(templi):
                    if li.get_linewidth() == 1.65:
                        indx5.append(i)
                for i5 in indx5[::-1]:
                    templi[i5].remove()

            self.fig = recsec(self.st_current, xlim=np.sort(self.xlims[-2:]),
                              ylim=ylims, scalfact=self.scalfact,
                              update=update, fighandle=self.fig, tscale=self.tscale,
                              norm=self.normflag, indfirst=self.indfirst,
                              maxtraces=self.maxtraces, textline=self.print1, textbox=True,
                              menu=self.menu_print, processing=self.processing_print,
                              quickdraw=self.quickdraw, vlines=self.vlines, picktimes=self.picktimes)

        try:
            self.fig.canvas.manager.window.raise_()
        except:
            print('could not raise window automatically, you must make the figure window active manually')

    def on_scroll(self, event):
        """
        scroll triggers
        """
        if self.maxtraces >= len(self.st):
            pass
        else:
            if event.button == 'down':  # view traces below
                self.indfirst = min(self.indfirst+1, len(self.st_current)-self.maxtraces)
                self.fig = recsec(self.st_current, xlim=np.sort(self.xlims[-2:]),
                                  scalfact=self.scalfact, update=True, tscale=self.tscale,
                                  fighandle=self.fig, norm=self.normflag,
                                  indfirst=self.indfirst, maxtraces=self.maxtraces,
                                  textline=self.print1, menu=self.menu_print,
                                  textbox=True, vlines=self.vlines, picktimes=self.picktimes)
            elif event.button == 'up':  # go back up to view other traces
                self.indfirst = max(self.indfirst-1, 0)
                self.fig = recsec(self.st_current, xlim=np.sort(self.xlims[-2:]),
                                  scalfact=self.scalfact, update=True, fighandle=self.fig,
                                  norm=self.normflag, indfirst=self.indfirst, tscale=self.tscale,
                                  maxtraces=self.maxtraces, textline=self.print1,
                                  menu=self.menu_print, textbox=True,
                                  vlines=self.vlines, picktimes=self.picktimes)


def nextpow2(val):
    """
    Find the next power of 2 from val
    """
    import math
    temp = math.floor(math.log(val, 2))
    return int(math.pow(2, temp+1))


def attach_distaz_IRIS(st, event_lat, event_lon):
    """
    attach a string of the distance,az,baz to st, uses IRIS webservices station tool
    """
    #from obspy.iris import Client
    #get station lat lon from database
    #client = Client()
    for i, trace in enumerate(st):
        if trace.stats.location == '':
            loc = '--'
        else:
            loc = trace.stats.location
        # build the url use to get station info from IRIS webservices
        url = ('http://service.iris.edu/fdsnws/station/1/query?net=%s&sta=%s&loc=%s&cha=%s&level=station&format=text&includecomments=true&nodata=404' % (trace.stats.network, trace.stats.station, loc, trace.stats.channel))
        with urllib.request.urlopen(url) as temp:
            file1 = temp.read().decode('utf-8')
        lines = [line.split('|') for line in file1.split('\n')[1:]]
        sta_lat = float(lines[0][2])
        sta_lon = float(lines[0][3])
        baz, az, dist = pyproj_distaz(sta_lat, sta_lon, event_lat, event_lon, ellps='WGS84')
        #result = client.distaz(sta_lat, sta_lon, event_lat, event_lon)
        trace.stats.rdist = dist  # result['distance']*111.32
        trace.stats.azimuth = az  # result['azimuth']
        trace.stats.back_azimuth = baz  # result['backazimuth']
        st[i] = trace
    return st


def attach_coords_IRIS(st):
    """
    attach coordinates to stations in Stream st by getting info from IRIS
    """
    for trace in st:
        # build the url use to get station info from IRIS webservices
        if trace.stats.location == '':
            loc = '--'
        else:
            loc = trace.stats.location
        url = ('http://service.iris.edu/fdsnws/station/1/query?net=%s&sta=%s&loc=%s&cha=%s&level=station&format=text&includecomments=true&nodata=404' % (trace.stats.network, trace.stats.station, loc, trace.stats.channel))
        with urllib.request.urlopen(url) as temp:
            file1 = temp.read().decode('utf-8')
        lines = [line.split('|') for line in file1.split('\n')[1:]]
        lat = float(lines[0][2])
        lon = float(lines[0][3])
        elev = float(lines[0][4])
        # add to trace
        trace.stats.coordinates = AttribDict({'latitude': lat, 'longitude': lon, 'elevation': elev})
    return st


def pyproj_distaz(lat1, lon1, lat2, lon2, ellps='WGS84'):
    """
    Find distance, azimuth, and backazimuth between two sets of coordinates
    USAGE
    pyproj_distaz(lat1, lon1, lat2, lon2, ellps='WGS84')
    INPUTS
    lat1 - latitude of point 1 in decimal degrees
    lon1 - longitude of point 1 in decimal degrees
    lat2 - latitude of point 2 in decimal degrees
    lon2 - longitude of point 2 in decimal degrees
    ellps = ellipsoid, WGS84 by default. help(Geod.__new__) gives a list of possible ellipsoids
    OUTPUTS
    az12 - azimuth from 1 to 2 in degrees (if 1 is station and 2 is source, this is backazimuth)
    az21 - azimuth from 2 to 1 in degrees (if 1 is station and 2 is source, this is azimuth)
    dist - great circle distance, in km
    """
    from pyproj import Geod
    g = Geod(ellps='WGS84')
    az12, az21, dist = g.inv(lon1, lat1, lon2, lat2)
    dist = dist/1000.
    if az12 < 0:
        az12 = 360+az12
    if az21 < 0:
        az21 = 360+az21
    return az12, az21, dist


def get_stations(event_lat, event_lon, event_time, clients=['IRIS'], minradiuskm=0., maxradiuskm=25,
                 chan='BH?,EH?,HH?,BDF', level='channel', **kwargs):
    """
    Wrapper around obspy get_stations function
    http://service.iris.edu/fdsnws/station/1/
    Can use ? wildcards in the channel designators, write whole channel list as one string
    startbefore = optional start time in case you want to use a different startbefore time than the event_time
    kwargs = any additional arguments from https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_stations.html

    Returns:
        station inventory
    """
    inventory = None
    for client in clients:
        client1 = FDSN_Client(client)
        try:
            inv = client1.get_stations(latitude=event_lat, longitude=event_lon, startbefore=event_time,
                                       endafter=event_time, minradius=minradiuskm/111.32,
                                       maxradius=maxradiuskm/111.32,
                                       channel=chan, level=level, **kwargs)
        except Exception:
            print('No stations found from %s Data Center' % (client,))
            inv = None
        if inventory is None:
            inventory = inv
        else:
            inventory += inv

    return inventory


def get_stations_iris(event_lat, event_lon, event_time, startbefore=None, minradiuskm=0., maxradiuskm=25, chan=('BH?,EH?,HH?,BDF')):
    """
    Get station info from IRIS webservices station tool for stations within specified radius
    http://service.iris.edu/fdsnws/station/1/
    Can use ? wildcards in the channel designators, write whole channel list as one string
    startbefore = optional start time in case you want to use a different startbefore time than the event_time
    """
    if startbefore is None:
        sttime = event_time.strftime('%Y-%m-%dT%H:%M:%S')
    else:
        sttime = startbefore.strftime('%Y-%m-%dT%H:%M:%S')
    # build the url use to get station info from IRIS webservices
    url = ('http://service.iris.edu/fdsnws/station/1/query?latitude=%f&longitude=%f&minradius=%f&maxradius=%f&cha=%s&startbefore=%s&endafter=%s&level=channel&format=text&nodata=404'
           % (event_lat, event_lon, minradiuskm/111.32, maxradiuskm/111.32, chan, sttime, event_time.strftime('%Y-%m-%dT%H:%M:%S')))
    with urllib.request.urlopen(url) as temp:
        file1 = temp.read().decode('utf-8')
    lines = [line.split('|') for line in file1.split('\n')[1:]]
    source = 'IRIS'
    return lines, source


def get_stations_ncedc(event_lat, event_lon, event_time, minradiuskm=0., maxradiuskm=25, chan=('BH?,EH?,HH?,BDF')):
    """
    Get station info from NCEDC webservices station tool for stations within specified radius
    service.ncedc.org/fdsnws/station/1/
    Can use ? wildcards in the channel designators, write whole channel list as one string
    """
    # build the url use to get station info from IRIS webservices
    url = ('http://service.ncedc.org/fdsnws/station/1/query?latitude=%f&longitude=%f&minradius=%f&maxradius=%f&cha=%s&startbefore=%s&endafter=%s&level=channel&format=text&nodata=404'
           % (event_lat, event_lon, minradiuskm/111.32, maxradiuskm/111.32, chan, event_time.strftime('%Y-%m-%dT%H:%M:%S'),
              event_time.strftime('%Y-%m-%dT%H:%M:%S')))
    with urllib.request.urlopen(url) as temp:
        file1 = temp.read().decode('utf-8')
    lines = [line.split('|') for line in file1.split('\n')[1:]]
    source = 'NCEDC'
    return lines, source


def getpeaks(st, pga=True, pgv=True, psa=True, periods=[0.3, 1.0, 3.0], damping=0.05, cosfilt=None, water_level=60.,
             csvfile=None, verbal=False, alreadycoor=False, corrto='acc', scale=1.):
    """
    Performs station correction (st must have response info attached to it or already be corrected) - removes trends and
    tapers with 5 percent cosine taper before doing station correction, adds as field in st and prints out results, option to save csv file
    All values in m/s and/or m/s**2
    USAGE

    INPUTS
    st - stream of obspy traces of raw seismic data with response information attached - best if visually inspected in case there are data problems
    pga - True if want to calculate pga
    pgv - True if want to calculate pgv
    psa - True if want to calculate peak spectral accelerations
    periods - periods at which to calculate psa
    damping - damping to use for psa calculations
    cosfilt - tuple of four corners, in Hz, for cosine filter to use in station correction. None for no cosine filter
    water_level - water level to use in station correction
    csvfile - full file path of csvfile to output with results, None if don't want to output csvfile
    verbal - if True, will print out all results to screen
    alreadycorr - if False, needs to be corrected for station response still and response info needs to be attached,
                    if True, need to set coorto to 'vel' or 'acc' and set scale to multiply units to get in terms of meters
    corrto - see alreadycoor
    scale - see alreadycoor

    OUTPUTS
    stacc - stream of data corrected to acceleration with pga's, pgv's and psa's attached, stored as AttribDict in in tr.stats.gmparam
    csvfile
    stvel - stream of data corrected to velocity with pga's, pgv's and psa's attached, stored as AttribDict in in tr.stats.gmparam
    """
    from obspy.core import AttribDict

    st.detrend('demean')
    st.detrend('linear')
    st.taper(max_percentage=0.05, type='cosine')

    # If coordinates aren't already attached, try to attach from IRIS
    if 'coordinates' not in st[0].stats:
        try:
            st = attach_coords_IRIS(st)  # Attach lats and lons if available
        except:
            print('Could not attach lats and lons, continuing')

    if alreadycoor:
        if corrto == 'acc':
            stacc = st.copy()
            stvel = st.copy().integrate()
        elif corrto == 'vel':
            stvel = st.copy()
            stacc = st.copy().differentiate()
        # Build place to store gm parameters
        for trace in stacc:
            trace.stats.gmparam = AttribDict()
        for trace in stvel:
            trace.stats.gmparam = AttribDict()

    else:
        stacc = st.copy()
        # Build place to store gm parameters
        for trace in stacc:
            trace.stats.gmparam = AttribDict()

        try:
            stacc.remove_response(output='ACC', pre_filt=cosfilt, water_level=water_level)
        except:
            print('Failed to do bulk station correction, trying one at a time')
            stacc = st.copy()  # Start with fresh data
            removeid = []
            for trace in stacc:
                try:
                    trace.remove_response(output='ACC', pre_filt=cosfilt, water_level=water_level)
                except:
                    print(('Failed to remove response for %s, deleting this station' % (trace.stats.station + trace.stats.channel,)))
                    removeid.append(trace.id)
            for rmid in removeid:  # Delete uncorrected ones
                for tr in stacc.select(id=rmid):
                    stacc.remove(tr)

        stvel = st.copy()
        # Build place to store gm parameters
        for trace in stvel:
            trace.stats.gmparam = AttribDict()
        try:
            stvel.remove_response(output='VEL', pre_filt=cosfilt, water_level=water_level)
        except:
            print('Failed to do bulk station correction, trying one at a time')
            stvel = st.copy()  # Start with fresh data
            removeid = []
            for trace in stvel:
                try:
                    trace.remove_response(output='VEL', pre_filt=cosfilt, water_level=water_level)
                except:
                    print(('Failed to remove response for %s, deleting this station' % (trace.stats.station + trace.stats.channel,)))
                    removeid.append(trace.id)
            for rmid in removeid:  # Delete uncorrected ones
                for tr in stvel.select(id=rmid):
                    stvel.remove(tr)

    if pga is True:
        for j, trace in enumerate(stacc):
            trace.stats.gmparam['pga'] = np.abs(trace.max())  # in obspy, max gives the max absolute value of the data
            stvel[j].stats.gmparam['pga'] = np.abs(trace.max())
            if verbal is True:
                print((('%s - PGA = %1.3f m/s') % (trace.id, np.abs(trace.max()))))

    if pgv is True:
        for j, trace in enumerate(stvel):
            trace.stats.gmparam['pgv'] = np.abs(trace.max())
            stacc[j].stats.gmparam['pgv'] = np.abs(trace.max())
            if verbal is True:
                print((('%s - PGV = %1.3f m/s') % (trace.id, np.abs(trace.max()))))

    if psa is True:
        for j, trace in enumerate(stacc):
            out = []
            for T in periods:
                freq = 1.0/T
                omega = (2 * 3.14159 * freq) ** 2
                paz_sa = corn_freq_2_paz(freq, damp=damping)
                paz_sa['sensitivity'] = omega
                paz_sa['zeros'] = []
                dd = simulate_seismometer(trace.data, trace.stats.sampling_rate, paz_remove=None, paz_simulate=paz_sa,
                                          taper=True, simulate_sensitivity=True, taper_fraction=0.05)
                if abs(max(dd)) >= abs(min(dd)):
                    psa1 = abs(max(dd))
                else:
                    psa1 = abs(min(dd))
                out.append(psa1)
                if verbal is True:
                    print((('%s - PSA at %1.1f sec = %1.3f m/s^2') % (trace.id, T, psa1)))
            trace.stats.gmparam['periods'] = periods
            trace.stats.gmparam['psa'] = out
            stvel[j].stats.gmparam['periods'] = periods
            stvel[j].stats.gmparam['psa'] = out

    if csvfile is not None:
        import csv
        with open(csvfile, 'wb') as csvfile1:
            writer = csv.writer(csvfile1)
            writer.writerow(['Id']+[tr.id for tr in st])
            try:
                #test = [tr.stats.coordinates['latitude'] for tr in st]
                writer.writerow(['Lat']+[tr.stats.coordinates['latitude'] for tr in st])
                writer.writerow(['Lon']+[tr.stats.coordinates['longitude'] for tr in st])
            except:
                print('Could not print out lats/lons to csvfile')
            if pga is True:
                writer.writerow(['PGA (m/s^2)']+[tr.stats.gmparam['pga'] for tr in stacc])
            if pgv is True:
                writer.writerow(['PGV (m/s)']+[tr.stats.gmparam['pgv'] for tr in stvel])
            if psa is True:
                for k, period in enumerate(periods):
                    writer.writerow(['PSA (m/s^2) at %1.1f sec, %1.0fpc damping' % (period, 100*damping)]+[tr.stats.gmparam['psa'][k] for tr in stacc])

    return stacc, stvel


def unique_list(seq):  # make a list only contain unique values and keep their order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def fourier_spectra(st, win=None, nfft=None, powerspec=False, recsec=False, xlim=None):
    """
    Plot multitaper fourier amplitude spectra of signals in st
    :param st: obspy stream or trace containing data to plot
    :param win: tuple of time window in seconds (e.g. win=(3., 20.)) over which to compute amplitude spectrum
    :param nfft: number of points to use in nfft, default None uses the next power of 2 of length of dat
    :param powerspec: = False for fourier amplitude spectrum, True for power spectrum
    :param recsec: if True, will plot spectra each in their own plot, otherwise all spectra will be in the same plot
    :param xlim: tuple of xlims like (0., 100.)
    """

    ###Should I have logx, logy as parameters?
    ###What about xunits? Won't it always be Freq? yunits?

    st = Stream(st)  # in case it's a trace
    st.detrend('demean')

    if recsec:
        fig, axes = plt.subplots(len(st), sharex=True, sharey=True, figsize=(10, 10))
    else:
        fig, ax = plt.subplots(1, figsize=(8, 5))

    plt.suptitle('Time period: %s to %s' % (str(st[0].stats.starttime), str(st[0].stats.endtime)))

    for i, st1 in enumerate(st):
        tvec = sigproc.maketvec(st1)  # Time vector
        dat = st1.data
        if win is not None:
            if win[1] > tvec.max() or win[0] < tvec.min():
                print('Time window specified not compatible with length of time series')
                return
            dat = dat[(tvec >= win[0]) & (tvec <= win[1])]
            tvec = tvec[(tvec >= win[0]) & (tvec <= win[1])]

        if nfft is None:
            nfft = int(nextpow2((st1.stats.endtime - st1.stats.starttime) * st1.stats. sampling_rate))
        st1.taper(max_percentage=0.05, type='cosine')

        if powerspec is False:
            amps = np.abs(np.fft.rfft(dat, n=nfft))
            freqs = np.fft.rfftfreq(nfft, 1/st1.stats.sampling_rate)
        else:
            amps = np.abs(np.fft.fft(dat, n=nfft))**2
            freqs = np.fft.fftfreq(nfft, 1/st1.stats.sampling_rate)
        if recsec:
            ax = axes[i]
        ax.plot(freqs, amps, label=st1.id)
        if i == len(st)-1 and recsec is False:
            plt.legend()
        elif recsec:
            plt.legend()
        if powerspec is False:
            plt.title('Amplitude Spectrum')
        else:
            plt.title('Power Spectrum')
    ax.set_xlabel('Frequency (Hz)')
    plt.show()
