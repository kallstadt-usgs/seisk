# seisk

## Introduction

seisk contains tools useful for interactively reviewing, manipulating, and processing seismic data. The software is composed of two main packages:

* reviewData - a wrapper for obspy that reads in data from IRIS, NCEDC, sac files, or winston waveservers and allows for interactive plotting, processing, and manipulation of this data. It also allows you to find stations within a given radius that were running at a given time and attach metadata such as lat\lon, azimuth, backazimuth, and source to station distance by calling IRIS webservices or NCEDC.
* sigproc - includes two modules, sigproc which contains signal processing functions and arrays, which contains array processing functions 

## Installation

### Install from scratch in a virtual environment
1. If a current version of conda is not already installed, install Miniconda 
    with Python 3.6 (or greater) following the directions provided on the 
    [conda webpage.](https://conda.io/docs/user-guide/install/index.html). 
    Anaconda will also work, but is a larger installation and is not necessary
    unless you want to use it for other purposes. Take note of the folder name
    where it is installed (e.g., miniconda or miniconda3)

2. The current version of miniconda requires that you manually edit your .bash_profile.
    Make the following changes, updating the path below with whatever folder miniconda was installed in:
    * If the installation added a line that looks like this, delete it:
        export PATH="/Users/YourName/miniconda3/bin:$PATH
    * add this line:
        . $HOME/miniconda3/etc/profile.d/conda.sh
    * Save and exit and either close the terminal and open a new one or source
        the .bash_profile ```source ~/.bash_profile```
    * Type ```which conda``` in terminal to make sure conda is found.

3. Clone the seisk repository in the location where you want it installed:
```sh
cd Users/YourName
git clone https://github.com/kallstadt-usgs/seisk.git
```
There will now be a folder called seisk in Users/YourName that contains
all of the files.

4. Run the install.sh script located in the main repository directory:
```sh
cd seisk
bash install.sh
```
This will take a while and will show numerous dependencies being installed.

5. The previous step installs a self-contained virtual environment called seisk.
    To ensure the virtual environment was successfully installed,
    type ```conda activate seisk```. You will need to activate the seisk environment
    every time you want to use these codes.

#### Updating

To ensure all of your dependencies are up to date, reinstall completely starting
at Step 3 above.

To update seisk to the current master branch without altering dependencies
(if you have altered the master branch, you will first need to stash your changes):
```sh
cd Users/YourName/seisk
git pull
```

#### Uninstalling

To uninstall, delete the virtual environment:
```sh
conda remove --name seisk --all
```
And remove the seisk folder that was cloned in step 3.

#### Troubleshooting

* Check step 2 from the installation steps above, make sure paths in .bash_profile are correct
    and point to the actual location of miniconda on your machine.

* Try opening a new terminal in case the updated .bash_profile was not sourced in the current terminal window.

* Uninstall (or move) your current anaconda or conda installation and reinstall from scratch. 
    Due to recent conda updates, older preexisting installations of anaconda or 
    miniconda may not function with our installer.
    
* Ensure that miniconda is in your user directory or somewhere that does not
    require admin permissions.

* If the entire virtual environment installation failed (install.sh), try running the following lines from within the top level of the seisk folder:
```
conda env create -f environment.yml --force
conda activate seisk
pip install -e .
```

### Install just this package and install dependencies separately

#### Initial install
pip install git+git://github.com/kallstadt-usgs/seisk.git

#### Upgrade
pip install -U git+git://github.com/kallstadt-usgs/seisk.git

#### Uninstall
pip uninstall seisk


## Interactive features
For the interactive plotting features to work (e.g., reviewData.InteractivePlot), the codes
must be run in ipython with pylab activated:
```
ipython --pylab
```

If you have a newer version of ipython, interactive plotting may still not work, try the following:
```
ipython matplotlib=qt
```

To make the qt backend the default backend for matplotlib, add the following line to your ~/.matplotlib/matplotlibrc file:
```
backend: Qt5Agg
```
