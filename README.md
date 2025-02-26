# ELEvoHI
ELlipse Evolution model based on Heliospheric Imager observations (ELEvoHI) models the evolution of coronal mass ejections and and predicts arrival times and speeds at any target in the heliosphere. Detailed information on the model can be found in T. Rollett et al. (2016, single-run version) and T. Amerstorfer et al. (2018, ensemble version).

In the following, it is described how ELEvoHI can be used based on data from heliospheric images alone, i.e. no coronagraph data are needed.

This is an ensemble model with the range of members being built around the defined input parameters phi (apex direction from HI observer), halfwidth (in degrees within the ecliptic plane) and inverse ellipse aspect ratio.

# How to use ELEvoHI

**1. Measure the CME front in heliospheric images**

The first and mandatory step to use ELEvoHI is to measure the propagating CME within heliospheric imager data. This can either be done in time-elongation maps or directly in the (running difference) images.
A Python-based tool for downloading HI data, data reduction and tracking can be found here:

https://github.com/maikebauer/STEREO-HI-Data-Processing

(Note: When you prefer to use tracks from a different source some paths need to be adjusted within the code.)

Usually, we measure a CME front at least 5 times and build a mean track in order to get a good representative of the CME evolution. This is already implemented in the model and done automatically after tracking.
Note, that the standard deviation derived from these measurements is not used within ELEvoHI.

**Important:** *Since ELEvoHI is used to predict arrival times and speeds at targets in the ecliptic, it is necessary to perform the measurements of the CME front within the ecliptic!*

**2. Cloning ELEvoHI**

Clone the ELEvoHI folder into the same directory where STEREO-HI-Data-Processing is located.

    git clone https://github.com/tamerstorfer/ELEvoHI.git
    cd ELEvoHI

**3. Installing requirements to run Python**

    conda create --n "helio_hi"
    conda activate helio_hi
    pip install -r requirements.txt
    cd code

**4. Setting up the ELEvoHI config file**

The config.json file is stored in the ELEvoHI/code folder and needs to be edited to set the parameter for the event of interest. When running ELEvoHI it is copied to the according folder of the event to save it.

In the config file, we need to define the following parameters:

    "basic_path": "/User/username/",
location of the folders STEREO-HI-Data-Processing and ELEvoHI.

    "eventdate": "20200623",
date of the first HI data point

    "HIobs": "A",
HI observer (A, B, PSP*, SolO*)
*to be implemented

    "mode": "science",
quality of HI data (beacon, science)

    "phi_FPF": true,
Set true if you want to run FPF in advance to derive the direction of motion.
Set false if you derive the direction from another source.

    "phi_manual": [71, 61, 81]
Propagation direction in degree of the CME apex relative to the HI observer. This parameter is ignored if FPF is set to true.
First entry is the direction for the deterministic run, second (third) entry is the lower (upper) value of the ensemble.

    "f": [0.8, 0.7, 0.9]
inverse ellipse aspect ratio (1 means circular, smaller values correspond to a flattened front)
First entry is the inverse aspect ratio for the deterministic run, second (third) entry is the lower (upper) value of the ensemble.

    "halfwidth": [39, 29, 49]
Halfwidth in degree within the ecliptic plane (minimum: 1, maximum: 90).
First entry is the halfwidth for the deterministic run, second (third) entry is the lower (upper) value of the ensemble.

    "startcut": 0,
When our STEREO-HI-Data-Processing tool is used to measure the CME front, the track consists of 30 time-elongation pairs.
Startcut corresponds to the first data point considered in the modelling.
Note: Since ELEvoHI is assuming a drag-based propagation, it is necessary that any early acceleration due to the Lorentz-force is excluded to be used for modelling. Usually, a reasonable intervall to perform the DBMfit is from 20 to 100 solar radii. For some events, the initial distance can be also very close to the Sun or even farther out. This depends on the kinematics of each event individually. 

    "endcut": 30,
Last data point cosidered for modelling. Use smaller values in case the DBMfit is not converging.
This might be the case for CMEs that are influenced by other CMEs or high-speed solar wind streams. To get a first glance of the CME kinematics, set the startcut to 0, the endcut to 30 and rerun if needed.

    "outer_system": false,
Set true for modelling up to the heliocentric distance of Neptune.
  
    "movie": false,
Set true to produce a movie. Make sure you have installed ffmpeg.

    "L1_ist_obs": "2020-06-30 01:12",
If known, you can state the detected in situ arrival time for L1 (or other targets). In this case, the difference of modelled and detected arrival time is directly calculated.
For realtime predictions delete this line.

    "L1_isv_obs": 332,
If known, you can state the detected in situ arrival speed (we usually use the mean speed in the sheath as a comparison).
For realtime predictions delete this line.

These are all available spacecraft and planets for which an arrival prediction can be performed.
State the measured in situ arrival time as follows:

    L1_ist_obs
    STEREOA_ist_obs
    STEREOB_ist_obs
    MESSENGER_ist_obs
    VENUSEXPRESS_ist_obs
    BEPICOLOMBO_ist_obs
    SOLARORBITER_ist_obs
    PARKERSOLARPROBE_ist_obs
    MERCURY_ist_obs
    VENUS_ist_obs
    EARTH_ist_obs
    MARS_ist_obs
    JUPITER_ist_obs
    SATURN_ist_obs
    URANUS_ist_obs
    NEPTUNE_ist_obs

State the measured in situ arrival speed as follows:

    L1_isv_obs
    STEREOA_isv_obs
    STEREOB_isv_obs
    MESSENGER_isv_obs
    VENUSEXPRESS_isv_obs
    BEPICOLOMBO_isv_obs
    SOLARORBITER_isv_obs
    PARKERSOLARPROBE_isv_obs
    MERCURY_isv_obs
    VENUS_isv_obs
    EARTH_isv_obs
    MARS_isv_obs
    JUPITER_isv_obs
    SATURN_isv_obs
    URANUS_isv_obs
    NEPTUNE_isv_obs

 In situ arrival times and speeds can be given independent from each other.

    "silent": true

If you want to avoid unnecessary terminal output, set this to true.

**5. Running ELEvoHI**
        
When the parameters needed are set and the helio_hi environment is active run ELEvoHI as follows:

    python ELEvoHI.py

When ELEvoHI has finished, the result of the prediction is given in the terminal. Plots and figures can be found in the according events folder under ELEvoHI/predictions.

The ensemble version of ELEvoHI (Amerstorfer et al., 2018) will soon follow in Python. Also, the possibility of frontal deformation (Hinterreiter et al., 2021b) will be implemented again in Python.
In case you find a bug or if you have any questions, please contact me (tanja.amerstorfer (at) geosphere.at).









