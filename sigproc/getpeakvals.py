from reviewData import reviewData
from obspy import UTCDateTime, read
from obspy.signal.invsim import seisSim, cornFreq2Paz

# Find stations near epicenter

event_lat = 22.83
event_lon = 120.625
event_time = UTCDateTime('2016-02-05T19:57:26')
t1 = event_time - 5.
t2 = event_time + 200.

lines, source = reviewData.get_stations_iris(event_lat, event_lon, event_time, minradiuskm=0., maxradiuskm=300., chan=('???'))
lines = lines[:-1]

netnames = reviewData.unique_list([line[0] for line in lines])
stanames = reviewData.unique_list([line[1] for line in lines])
channels = '?N?,?L?'  # findsta.unique_list([line[3] for line in lines])
location = '*'

st = reviewData.getdata(','.join(reviewData.unique_list(netnames)), ','.join(reviewData.unique_list(stanames)), location, channels, t1, t2, attach_response=True, savedat=True, folderdat='data', filenamepref='Data_', clientname='IRIS', loadfromfile=False)

st.detrend('demean')
st.detrend('linear')

cosfilt = None  # (0.01, 0.02, 20, 30)
output = 'ACC'

# Press c when in interactive mode to correct for station response, then use A to select amplitude picks
zp = reviewData.InteractivePlot(st, cosfilt=cosfilt, output=output)

pickdataPGA = zp.picks
for key in pickdataPGA:
    print('%s - PGA = %1.2f m/s^2') % (pickdataPGA[key]['stachan'], pickdataPGA[key]['weight'])

# Get PSA
stacc = zp.st_current.copy()
damping = 0.05
periods = [0.3, 1.0, 3.0]

for tr in stacc:
    out = []
    for T in periods:
        freq = 1.0/T
        omega = (2 * 3.14159 * freq) ** 2
        paz_sa = cornFreq2Paz(freq, damp=damping)
        paz_sa['sensitivity'] = omega
        paz_sa['zeros'] = []
        dd = seisSim(tr.data, tr.stats.sampling_rate, paz_remove=None, paz_simulate=paz_sa,
                     taper=True, simulate_sensitivity=True, taper_fraction=0.05)
        if abs(max(dd)) >= abs(min(dd)):
            psa = abs(max(dd))
        else:
            psa = abs(min(dd))
        out.append(psa)
        print('%s - PSA at %1.1f sec = %1.3f m/s^2') % (tr.id, T, psa)

# Now get PGV

output = 'VEL'
zp = reviewData.InteractivePlot(st, cosfilt=cosfilt, output=output)

pickdataPGV = zp.picks
for key in pickdataPGV:
    print('%s - PGV = %1.3f m/s') % (pickdataPGV[key]['stachan'], pickdataPGV[key]['weight'])
