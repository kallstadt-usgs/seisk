#reviewData

reviewData has two parts:
*reviewData.reviewData - a wrapper for obspy that reads in data from IRIS, NCEDC, sac files, or winston waveservers and allows for interactive plotting, processing, and manipulation of this data. It also allows you to find stations within a given radius that were running at a given time and attach metadata such as lat\lon, azimuth, backazimuth, and source to station distance by calling IRIS webservices or NCEDC.
*reviewData.sigproc - some random signal processing codes

##Installation
Do this in your home directory. This is assuming you use Anaconda and have done conda install pip. I have no idea how to install it otherwise other than adding it to your python path directly. Good luck.

###Initial install
pip install git+git://github.com/kallstadt-usgs/seisk.git

###Upgrade
pip install -U git+git://github.com/kallstadt-usgs/seisk.git

###Uninstall
pip uninstall seisk

###Dependencies
####Things that come with Anaconda (I think)
numpy
scipy
matplotlib
os
textwrap
urllib2
####Other
#####obspy
[ObsPy](https://github.com/obspy/obspy/wiki) can be installed using Anaconda:
conda install -c obspy obspy
