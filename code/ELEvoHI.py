
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from datetime import datetime
from datetime import timedelta
import matplotlib.dates as mdates
from matplotlib.pyplot import cm
from astropy.time import Time as atime
from astropy import constants as const
from sunpy.coordinates import frames, get_horizons_coord
import csv
import pandas as pd
import seaborn as sns
import json
import pickle
import csv
import logging
import shutil

from functions import load_config, calculate_new_time_axis, merge_tracks, fpf_function, fpf, ELCon, fitdbm, fitdbmneg, cost_function, cost_functionneg, DBMfitting, elevo_analytic, elevo, assess_prediction

def main():
    #global tinit, rinit, vinit, swspeed, xdata, ydata, direction, prediction_path
    global prediction_path

    # Constants
    AU = const.au.to_value('km')
    rsun = const.R_sun.to_value('km')

    config = load_config('config.json')

    basic_path = config['basic_path']
    pred_path = basic_path + 'ELEvoHI/predictions/'
    eventdate = config['eventdate']
    HIobs = config['HIobs']
    mode = config['mode']
    phi_FPF = config['phi_FPF']
    phi_manual = config['phi_manual']
    f = config['f']
    halfwidth = np.deg2rad(config['halfwidth'])
    startcut = config['startcut']
    endcut = config['endcut']
    outer_system = config['outer_system']
    movie = config['movie']
    silent = config['silent']
    
    if not os.path.exists(pred_path):
        os.mkdir(pred_path)
    
    L1_istime = config.get('L1_ist_obs', None)
    if not L1_istime == None:
        L1_istime = datetime.strptime(L1_istime, "%Y-%m-%d %H:%M")
    L1_isspeed = config.get('L1_isv_obs', np.nan)
    
    STEREOA_istime = config.get('STEREOA_ist_obs', None)
    if not STEREOA_istime == None:
        STEREOA_istime = datetime.strptime(STEREOA_istime, "%Y-%m-%d %H:%M")       
    STEREOA_isspeed = config.get('STEREOA_isv_obs', np.nan)
    
    STEREOB_istime = config.get('STEREOB_ist_obs', None)
    if not STEREOB_istime == None:
        STEREOB_istime = datetime.strptime(STEREOB_istime, "%Y-%m-%d %H:%M")
    STEREOB_isspeed = config.get('STEREOB_isv_obs', np.nan)
    
    MES_istime = config.get('MESSENGER_ist_obs', None)
    if not MES_istime == None:
        MES_istime = datetime.strptime(MES_istime, "%Y-%m-%d %H:%M")
    MES_isspeed = config.get('MESSENGER_isv_obs', np.nan)
    
    VEX_istime = config.get('VENUSEXPRESS_ist_obs', None)
    if not VEX_istime == None:
        VEX_istime = datetime.strptime(VEX_istime, "%Y-%m-%d %H:%M")
    VEX_isspeed = config.get('VENUSEXPRESS_isv_obs', np.nan)
    
    BEPI_istime = config.get('BEPICOLOMBO_ist_obs', None)
    if not BEPI_istime == None:
        BEPI_istime = datetime.strptime(BEPI_istime, "%Y-%m-%d %H:%M")
    BEPI_isspeed = config.get('BEPICOLOMBO_isv_obs', np.nan)
    
    SOLO_istime = config.get('SOLARORBITER_ist_obs', None)
    if not SOLO_istime == None:
        SOLO_istime = datetime.strptime(SOLO_istime, "%Y-%m-%d %H:%M")
    SOLO_isspeed = config.get('SOLARORBITER_isv_obs', np.nan)
    
    PSP_istime = config.get('PARKERSOLARPROBE_ist_obs', None)
    if not PSP_istime == None:
        PSP_istime = datetime.strptime(PSP_istime, "%Y-%m-%d %H:%M")
    PSP_isspeed = config.get('PARKERSOLARPROBE_isv_obs', np.nan)
    
    MERCURY_istime = config.get('MERCURY_ist_obs', None)
    if not MERCURY_istime == None:
        MERCURY_istime = datetime.strptime(MERCURY_istime, "%Y-%m-%d %H:%M")
    MERCURY_isspeed = config.get('MERCURY_isv_obs', np.nan)
    
    VENUS_istime = config.get('VENUS_ist_obs', None)
    if not VENUS_istime == None:
        VENUS_istime = datetime.strptime(VENUS_istime, "%Y-%m-%d %H:%M")
    VENUS_isspeed = config.get('VENUS_isv_obs', np.nan)
    
    EARTH_istime = config.get('EARTH_ist_obs', None)
    if not EARTH_istime == None:
        EARTH_istime = datetime.strptime(EARTH_istime, "%Y-%m-%d %H:%M")
    EARTH_isspeed = config.get('EARTH_isv_obs', np.nan)
    
    MARS_istime = config.get('MARS_ist_obs', None)
    if not MARS_istime == None:
        MARS_istime = datetime.strptime(MARS_istime, "%Y-%m-%d %H:%M")
    MARS_isspeed = config.get('MARS_isv_obs', np.nan)

    if outer_system:
        JUPITER_istime = config.get('JUPITER_ist_obs', None)
        if not JUPITER_istime == None:
            JUPITER_istime = datetime.strptime(JUPITER_istime, "%Y-%m-%d %H:%M")
        JUPITER_isspeed = config.get('JUPITER_isv_obs', np.nan)
        
        SATURN_istime = config.get('SATURN_ist_obs', None)
        if not SATURN_istime == None:
            SATURN_istime = datetime.strptime(SATURN_istime, "%Y-%m-%d %H:%M")
        SATURN_isspeed = config.get('SATURN_isv_obs', np.nan)
        
        URANUS_istime = config.get('URANUS_ist_obs', None)
        if not URANUS_istime == None:
            URANUS_istime = datetime.strptime(URANUS_istime, "%Y-%m-%d %H:%M")
        URANUS_isspeed = config.get('URANUS_isv_obs', np.nan)
        
        NEPTUNE_istime = config.get('NEPTUNE_ist_obs', None)
        if not NEPTUNE_istime == None:
            NEPTUNE_istime = datetime.strptime(NEPTUNE_istime, "%Y-%m-%d %H:%M")
        NEPTUNE_isspeed = config.get('NEPTUNE_isv_obs', np.nan)
    
    year = eventdate[:4]
    event_path = basic_path + 'STEREO-HI-Data-Processing/data/stereo_processed/jplot/' + HIobs + '/' + mode + '/hi1hi2/' + year + '/Tracks/' + eventdate + '/'
    prediction_path = pred_path + eventdate + '_' + HIobs + '/'
    
    # combines the time-elongation tracks into one average track on a equitemporal time-axis
    # includes standard deviation and saves a figure to the predictions folder
    track = merge_tracks(event_path, prediction_path)
    
    shutil.copy(basic_path + 'ELEvoHI/code/config.json', prediction_path)
    
    if phi_FPF:
        fpf_fit = fpf(track, startcut, endcut, prediction_path)
        phi = np.deg2rad(fpf_fit['phi_FPF'])
    else:
        phi = np.deg2rad(phi_manual)
        
    elon = track["elongation"]

    # Convert the "time" column to datetime objects
    track["time"] = pd.to_datetime(track["time"])

    time = track["time"]

    # set the initial time to the first measurement from HI observations
    thi = time[0]

    # Convert datetime values to seconds since the first element
    time_num = (time - time.iloc[0]).dt.total_seconds()
    
    #####################################################################
    # get s/c and planet positions

    #STEREO Ahead
    start_date = datetime(2006, 10, 26, 0, 2)
    
    logging.getLogger('sunpy.coordinates').setLevel(logging.WARNING)

    if start_date < thi:
        coord = get_horizons_coord('STEREO-A', thi)
        sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
        sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ

        sta_time = sc_hee.obstime.to_datetime()
        sta_r = sc_heeq.radius.value
        sta_lon = np.deg2rad(sc_hee.lon.value)
        sta_lat = np.deg2rad(sc_hee.lat.value)
        sta_available = 1
        if HIobs == 'A':
            d = sta_r
            if sta_lon >=0:
                direction = sta_lon - phi
            else:
                direction = sta_lon + phi       
    else: sta_available = 0

    # Parker Solar Probe
    start_date = datetime(2018, 8, 12, 9, 0)

    if start_date < thi:
        coord = get_horizons_coord('Parker Solar Probe', thi)  
        sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
        sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
        psp_time = sc_hee.obstime.to_datetime()
        psp_r = sc_heeq.radius.value
        psp_lon = np.deg2rad(sc_hee.lon.value)
        psp_lat = np.deg2rad(sc_hee.lat.value)
        psp_available = 1
    else: psp_available = 0
        
    # Solar Orbiter    
    start_date = datetime(2020, 2, 10, 5, 0)

    if start_date < thi:
        coord = get_horizons_coord('Solar Orbiter', thi)
        sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
        sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
        solo_time = sc_hee.obstime.to_datetime()
        solo_r = sc_heeq.radius.value
        solo_lon = np.deg2rad(sc_hee.lon.value)
        solo_lat = np.deg2rad(sc_hee.lat.value)
        solo_available = 1
    else: solo_available = 0

    # BepiColombo
    start_date = datetime(2018, 10, 20, 3, 0)

    if start_date < thi:

        coord = get_horizons_coord('BepiColombo', thi)
        sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
        sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
        bepi_time = sc_hee.obstime.to_datetime()
        bepi_r = sc_heeq.radius.value
        bepi_lon = np.deg2rad(sc_hee.lon.value)
        bepi_lat = np.deg2rad(sc_hee.lat.value)
        bepi_available = 1
    else: bepi_available = 0
        
    # STEREO Behind
    start_date = datetime(2006, 10, 26, 2, 0)
    end_date = datetime(2014, 9, 30, 23, 0)

    if start_date < thi < end_date:
        coord = get_horizons_coord('STEREO-B', thi)
        sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
        sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
        stb_time = sc_hee.obstime.to_datetime()
        stb_r = sc_heeq.radius.value
        stb_lon = np.deg2rad(sc_hee.lon.value)
        stb_lat = np.deg2rad(sc_hee.lat.value)
        stb_available = 1
        if HIobs == 'B':
            d = stb_r
            if stb_lon >=0:
                direction = stb_lon - phi
            else:
                direction = stb_lon + phi
    else: stb_available = 0
        
    # Venus Express
    start_date = datetime(2005, 11, 9, 6, 0)
    end_date = datetime(2014, 12, 31, 23, 0)

    if start_date < thi < end_date:
        coord = get_horizons_coord('Venus Express', thi)
        sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
        sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
        vex_time = sc_hee.obstime.to_datetime()
        vex_r = sc_heeq.radius.value
        vex_lon = np.deg2rad(sc_hee.lon.value)
        vex_lat = np.deg2rad(sc_hee.lat.value)
        vex_available = 1
    else: vex_available = 0
        
    # Messenger
    start_date = datetime(2004, 8, 3, 8, 0)
    end_date = datetime(2015, 5, 1, 18, 0)

    if start_date < thi < end_date:
        coord = get_horizons_coord('Messenger', thi)
        sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
        sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
        mes_time = sc_hee.obstime.to_datetime()
        mes_r = sc_heeq.radius.value
        mes_lon = np.deg2rad(sc_hee.lon.value)
        mes_lat = np.deg2rad(sc_hee.lat.value)
        mes_available = 1
    else: mes_available = 0

    # Mercury
    coord = get_horizons_coord(199, thi)
    sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
    sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
    mercury_time = sc_hee.obstime.to_datetime()
    mercury_r = sc_heeq.radius.value
    mercury_lon = np.deg2rad(sc_hee.lon.value)
    mercury_lat = np.deg2rad(sc_hee.lat.value)

    # Venus
    coord = get_horizons_coord(299, thi)
    sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
    sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
    venus_time = sc_hee.obstime.to_datetime()
    venus_r = sc_heeq.radius.value
    venus_lon = np.deg2rad(sc_hee.lon.value)
    venus_lat = np.deg2rad(sc_hee.lat.value)

    # L1
    coord = get_horizons_coord('EM-L1', thi)
    sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
    sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
    L1_time = sc_hee.obstime.to_datetime()
    L1_r = sc_heeq.radius.value
    L1_lon = np.deg2rad(sc_hee.lon.value)
    L1_lat = np.deg2rad(sc_hee.lat.value)

    # Earth
    coord = get_horizons_coord(399, thi)
    sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
    sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
    earth_time = sc_hee.obstime.to_datetime()
    earth_r = sc_heeq.radius.value
    earth_lon = np.deg2rad(sc_hee.lon.value)
    earth_lat = np.deg2rad(sc_hee.lat.value)

    # Mars
    coord = get_horizons_coord(499, thi)
    sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
    sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
    mars_time = sc_hee.obstime.to_datetime()
    mars_r = sc_heeq.radius.value
    mars_lon = np.deg2rad(sc_hee.lon.value)
    mars_lat = np.deg2rad(sc_hee.lat.value)

    if outer_system == 1:
        # Jupiter
        coord = get_horizons_coord(599, thi)
        sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
        sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ

        jupiter_time = sc_hee.obstime.to_datetime()
        jupiter_r = sc_heeq.radius.value
        jupiter_lon = np.deg2rad(sc_hee.lon.value)
        jupiter_lat = np.deg2rad(sc_hee.lat.value)

        # Saturn
        coord = get_horizons_coord(699, thi)
        sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
        sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ

        saturn_time = sc_hee.obstime.to_datetime()
        saturn_r = sc_heeq.radius.value
        saturn_lon = np.deg2rad(sc_hee.lon.value)
        saturn_lat = np.deg2rad(sc_hee.lat.value)

        # Uranus
        coord = get_horizons_coord(799, thi)
        sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
        sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ

        uranus_time = sc_hee.obstime.to_datetime()
        uranus_r = sc_heeq.radius.value
        uranus_lon = np.deg2rad(sc_hee.lon.value)
        uranus_lat = np.deg2rad(sc_hee.lat.value)

        # Neptune
        coord = get_horizons_coord(899, thi)
        sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
        sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ

        neptune_time = sc_hee.obstime.to_datetime()
        neptune_r = sc_heeq.radius.value
        neptune_lon = np.deg2rad(sc_hee.lon.value)
        neptune_lat = np.deg2rad(sc_hee.lat.value)
        
    availability = {
        'sta_available': sta_available,
        'stb_available': stb_available,
        'mes_available': mes_available,
        'vex_available': vex_available,
        'solo_available': solo_available,
        'psp_available': psp_available,
        'bepi_available': bepi_available
    }
    
    positions = {
        'L1_lon': L1_lon,
        'mercury_lon': mercury_lon,
        'venus_lon': venus_lon,
        'earth_lon': earth_lon,
        'mars_lon': mars_lon,
        'L1_r': L1_r,
        'mercury_r': mercury_r,
        'venus_r': venus_r,
        'earth_r': earth_r,
        'mars_r': mars_r,

    }
    
    if stb_available:
        # New entry as a dictionary
        add_stb = {'stb_lon': stb_lon,
               'stb_r': stb_r
        }
        # Adding the new entry using update()
        positions.update(add_stb)
    if sta_available:
        # New entry as a dictionary
        add_sta = {'sta_lon': sta_lon,
               'sta_r': sta_r
        }
        # Adding the new entry using update()
        positions.update(add_sta)
    if mes_available:
        # New entry as a dictionary
        add_mes = {'mes_lon': mes_lon,
               'mes_r': mes_r
        }
        # Adding the new entry using update()
        positions.update(add_vex)
    if vex_available:
        # New entry as a dictionary
        add_vex = {'vex_lon': vex_lon,
               'vex_r': vex_r
        }
        # Adding the new entry using update()
        positions.update(add_solo) 
    if solo_available:
        # New entry as a dictionary
        add_solo = {'solo_lon': solo_lon,
               'solo_r': solo_r
        }
        # Adding the new entry using update()
        positions.update(add_solo)
    if psp_available:
        # New entry as a dictionary
        add_psp = {'psp_lon': psp_lon,
               'psp_r': psp_r
        }
        # Adding the new entry using update()
        positions.update(add_psp)  
    if bepi_available:
        # New entry as a dictionary
        add_bepi = {'bepi_lon': bepi_lon,
               'bepi_r': bepi_r
        }
        # Adding the new entry using update()
        positions.update(add_bepi)
    if outer_system:
        outer_planets = {'jupiter_r': jupiter_r,
                         'saturn_r': saturn_r,
                         'uranus_r': uranus_r,
                         'neptune_r': neptune_r,
                         'jupiter_lon': jupiter_lon,
                         'saturn_lon': saturn_lon,
                         'uranus_lon': uranus_lon,
                         'neptune_lon': neptune_lon
        }
        positions.update(outer_planets)
    print(positions)
              
    ############################################################################   
    # calculate angular separation of CME apex from each target

    if abs(direction) + abs(mars_lon) < np.pi:
        delta_mars = direction - mars_lon
    else:
        delta_mars = direction - (mars_lon + 2 * np.pi * np.sign(direction))

    if abs(direction) + abs(venus_lon) < np.pi:
        delta_venus = direction - venus_lon
    else:
        delta_venus = direction - (venus_lon + 2 * np.pi * np.sign(direction))

    if abs(direction) + abs(mercury_lon) < np.pi:
        delta_mercury = direction - mercury_lon
    else:
        delta_mercury = direction - (mercury_lon + 2 * np.pi * np.sign(direction))
        
    if abs(direction) + abs(L1_lon) < np.pi:
        delta_L1 = direction - L1_lon
    else:
        delta_L1 = direction - (L1_lon + 2 * np.pi * np.sign(direction))

    if abs(direction) + abs(earth_lon) < np.pi:
        delta_earth = direction - earth_lon
    else:
        delta_earth = direction - (earth_lon + 2 * np.pi * np.sign(direction))

    delta_values = {
        'delta_L1': delta_L1,
        'delta_mercury': delta_mercury,
        'delta_venus': delta_venus,
        'delta_earth': delta_earth,
        'delta_mars': delta_mars,
    }

    if sta_available:
        if abs(direction) + abs(sta_lon) < np.pi:
            delta_sta = direction - sta_lon
            # New entry as a dictionary
        else:
            delta_sta = direction - (sta_lon + 2 * np.pi * np.sign(direction))
        add_sta = {'delta_sta': delta_sta}
        delta_values.update(add_sta)

    if psp_available:
        if abs(direction) + abs(psp_lon) < np.pi:
            delta_psp = direction - psp_lon
        else:
            delta_psp = direction - (psp_lon + 2 * np.pi * np.sign(direction))
        add_psp = {'delta_psp': delta_psp}
        delta_values.update(add_psp)

    if solo_available:
        if abs(direction) + abs(solo_lon) < np.pi:
            delta_solo = direction - solo_lon
        else:
            delta_solo = direction - (solo_lon + 2 * np.pi * np.sign(direction))
        add_solo = {'delta_solo': delta_solo}
        delta_values.update(add_solo)

    if bepi_available:
        if abs(direction) + abs(bepi_lon) < np.pi:
            delta_bepi = direction - bepi_lon
        else:
            delta_bepi = direction - (bepi_lon + 2 * np.pi * np.sign(direction))
        add_bepi = {'delta_bepi': delta_bepi}
        delta_values.update(add_bepi)

    if stb_available:
        if abs(direction) + abs(stb_lon) < np.pi:
            delta_stb = direction - stb_lon
        else:
            delta_stb = direction - (stb_lon + 2 * np.pi * np.sign(direction))
        add_stb = {'delta_stb': delta_stb}
        delta_values.update(add_stb)

    if vex_available:
        if abs(direction) + abs(vex_lon) < np.pi:
            delta_vex = direction - vex_lon
        else:
            delta_vex = direction - (vex_lon + 2 * np.pi * np.sign(direction))
        add_vex = {'delta_vex': delta_vex}
        delta_values.update(add_vex)
            
    if mes_available:
        if abs(direction) + abs(mes_lon) < np.pi:
            delta_mes = direction - mes_lon
        else:
            delta_mes = direction - (mes_lon + 2 * np.pi * np.sign(direction))
        add_mes = {'delta_mes': delta_mes}
        delta_values.update(add_mes)

    if outer_system == 1:
        if abs(direction) + abs(jupiter_lon) < np.pi:
            delta_jupiter = direction - jupiter_lon
        else:
            delta_jupiter = direction - (jupiter_lon + 2 * np.pi * np.sign(direction))

        if abs(direction) + abs(saturn_lon) < np.pi:
            delta_saturn = direction - saturn_lon
        else:
            delta_saturn = direction - (saturn_lon + 2 * np.pi * np.sign(direction))

        if abs(direction) + abs(uranus_lon) < np.pi:
            delta_uranus = direction - uranus_lon
        else:
            delta_uranus = direction - (uranus_lon + 2 * np.pi * np.sign(direction))
            
        if abs(direction) + abs(neptune_lon) < np.pi:
            delta_neptune = direction - neptune_lon
        else:
            delta_neptune = direction - (neptune_lon + 2 * np.pi * np.sign(direction))
        delta_outer = {
            'delta_jupiter': delta_jupiter,
            'delta_saturn': delta_saturn,
            'delta_uranus': delta_uranus,
            'delta_neptune': delta_neptune}
        delta_values.update(delta_outer)
            
    #####################################################################################
    # print deltas for each target and check for hits

    if sta_available == 1:
        if silent == 0:
            print('Delta(CME apex, STEREO-A): ', np.rad2deg(delta_sta))
        if np.abs(delta_sta) < halfwidth:
            hit_sta = 1
        else:
            hit_sta = 0
    else:
        hit_sta = 0

    if psp_available == 1:
        if silent == 0:
            print('Delta(CME apex, Parker Solar Probe): ', np.rad2deg(delta_psp))
        if np.abs(delta_psp) < halfwidth:
            hit_psp = 1
        else:
            hit_psp = 0
    else:
        hit_psp = 0

    if solo_available == 1:
        if silent == 0:
            print('Delta(CME apex, SolO): ', np.rad2deg(delta_solo))
        if np.abs(delta_solo) < halfwidth:
            hit_solo = 1
        else:
            hit_solo = 0
    else:
        hit_solo = 0

    if bepi_available == 1:
        if silent == 0:
            print('Delta(CME apex, BepiColombo): ', np.rad2deg(delta_bepi))
        if np.abs(delta_bepi) < halfwidth:
            hit_bepi = 1
        else:
            hit_bepi = 0
    else:
        hit_bepi = 0

    if stb_available == 1:
        if silent == 0:
            print('Delta(CME apex, STEREO-B): ', np.rad2deg(delta_stb))
        if np.abs(delta_stb) < halfwidth:
            hit_stb = 1
        else:
            hit_stb = 0
    else:
        hit_stb = 0

    if vex_available == 1:
        if silent == 0:
            print('Delta(CME apex, Venus Express): ', np.rad2deg(delta_vex))
        if np.abs(delta_vex) < halfwidth:
            hit_vex = 1
        else:
            hit_vex = 0
    else:
        hit_vex = 0
            
    if mes_available == 1:
        if silent == 0:
            print('Delta(CME apex, MESSENGER: ', np.rad2deg(delta_mes))
        if np.abs(delta_mes) < halfwidth:
            hit_mes = 1
        else:
            hit_mes = 0
    else:
        hit_mes = 0
        
    if silent == 0:
            print('Delta(CME apex, Mercury): ', np.rad2deg(delta_mercury))

    if np.abs(delta_mercury) < halfwidth:
        hit_mercury = 1
    else:
        hit_mercury = 0
        
    if silent == 0:
            print('Delta(CME apex, Venus): ', np.rad2deg(delta_venus))

    if np.abs(delta_venus) < halfwidth:
        hit_venus = 1
    else:
        hit_venus = 0

    if silent == 0:
            print('Delta(CME apex, L1): ', np.rad2deg(delta_L1))

    if np.abs(delta_L1) < halfwidth:
        hit_L1 = 1
    else:
        hit_L1 = 0

    if silent == 0:
            print('Delta(CME apex, Earth): ', np.rad2deg(delta_earth))

    if np.abs(delta_earth) < halfwidth:
        hit_earth = 1
    else:
        hit_earth = 0

    if silent == 0:
            print('Delta(CME apex, Mars): ', np.rad2deg(delta_mars))

    if np.abs(delta_mars) < halfwidth:
        hit_mars = 1
    else:
        hit_mars = 0
    
    hit_counts = {
        'hit_sta': hit_sta,
        'hit_stb': hit_stb,
        'hit_mes': hit_mes,
        'hit_vex': hit_vex,
        'hit_solo': hit_solo,
        'hit_psp': hit_psp,
        'hit_bepi': hit_bepi,
        'hit_L1': hit_L1,
        'hit_mercury': hit_mercury,
        'hit_venus': hit_venus,
        'hit_earth': hit_earth,
        'hit_mars': hit_mars,
    }

    if outer_system == 1:
        
        if silent == 0:
            print('Delta(CME apex, Jupiter): ', np.rad2deg(delta_jupiter))

        if np.abs(delta_jupiter) < halfwidth:
            hit_jupiter = 1
        else:
            hit_jupiter = 0
            
        if silent == 0:
            print('Delta(CME apex, Saturn): ', np.rad2deg(delta_saturn))

        if np.abs(delta_saturn) < halfwidth:
            hit_saturn = 1
        else:
            hit_saturn = 0
            
        if silent == 0:
            print('Delta(CME apex, Uranus): ', np.rad2deg(delta_uranus))

        if np.abs(delta_uranus) < halfwidth:
            hit_uranus = 1
        else:
            hit_uranus = 0
        
        if silent == 0:
            print('Delta(CME apex, Neptune): ', np.rad2deg(delta_neptune))

        if np.abs(delta_neptune) < halfwidth:
            hit_neptune = 1
        else:
            hit_neptune = 0
        outer_hits = {
            'hit_jupiter': hit_jupiter,
            'hit_saturn': hit_saturn,
            'hit_uranus': hit_uranus,
            'hit_neptune': hit_neptune
        }
            
    ########################################################################
    # convert the time-elongation track to radial distance using ELCon
    
    elon_rad = np.deg2rad(elon)
    R_elcon = ELCon(elon_rad, d, phi, halfwidth, f)   
    
    ########################################################################
    # run DBMfitting    
    
    gamma_valid, winds_valid, res_valid, tinit, rinit, vinit, swspeed, xdata, ydata = DBMfitting(time, R_elcon, prediction_path, startfit=startcut, endfit=endcut)
    
    # make equidistant grid for ELEvo times, with 10 min resolution

    start_time = tinit

    # Define time step as a timedelta object
    time_step = timedelta(minutes=10)

    # Define number of time steps
    if outer_system == 1:
        timegrid = 11000
    else:
        timegrid = 1440

    # Generate time array
    time_array = [start_time + i * time_step for i in range(timegrid)]
    
    # create 1-D DBM kinematic for ellipse apex with
    # constant drag parameter and constant background solar wind speed

    gamma = gamma_valid[0]
    winds = winds_valid[0]

    # numeric time axis starting at zero
    tnum = [(time_array[i] - start_time).total_seconds() for i in range(timegrid)]
    # speed array
    vdrag = np.zeros(timegrid, dtype=float)
    # distance array
    rdrag = np.zeros(timegrid, dtype=float)

    # then use Vrsnak et al. 2013 equation 5 for v(t), 6 for r(t)

    # acceleration or deceleration
    # Note that the sign in the dbm equation is related to vinit and the ambient wind speed.

    if vinit < winds:
        accsign = -1
        print('negative')
    else:
        accsign = 1
        print('positive')

    for i in range(timegrid):
        # heliocentric distance of CME apex in km
        rdrag[i] = (accsign / (gamma)) * np.log(1 + (accsign * (gamma) * ((vinit - winds) * tnum[i]))) + winds * tnum[i] + rinit
        # speed in km/s
        vdrag[i] = (vinit - winds) / (1 + (accsign * (gamma) * ((vinit - winds) * tnum[i]))) + winds

        if not np.isfinite(rdrag[i]):
            print('Sign of gamma does not fit to vinit and w!')
            raise ValueError('Invalid value in rdrag')
    
    ########################################################################
    # run the final prediction using ELEvo   
    
    R = rdrag /AU

    prediction = elevo(R, time_array, tnum, direction, f, halfwidth, vdrag, track, availability, hit_counts, delta_values, positions, HIobs, outer_system, prediction_path, movie=movie, timegrid=timegrid)
           
    if accsign < 0:
        print('')
        print('The CME is accelerated by the ambient solar wind.')
    else:
        print('')
        print('The CME is decelerated by the ambient solar wind.')

    formatted_drag = "{:.2f}".format(gamma*1e7)
        
    print('Drag parameter: ', formatted_drag, 'e-7 /km')
    print('Ambient solar wind speed: ', winds, 'km/s')
    print('Elongation range used for prediction:', round(elon[startcut], 1), '-', round(elon[endcut-1], 1),' degree')
    print('Corresponding heliocentric distance [AU]:', round(R_elcon[startcut], 2), '-', round(R_elcon[endcut-1], 2),' AU')
    print('Corresponding heliocentric distance [Rsun]:', round(R_elcon[startcut]*AU/rsun, 1), '-', round(R_elcon[endcut-1]*AU/rsun, 1),' Rsun')
    print('')
    print('Drag parameter:', round(gamma*1e7, 2), '1e-7')
    print('Ambient solar wind speed:', winds, 'km/s')
    
    # Check if "L1" is present in the 'target' column
    target_l1_present = prediction['target'] == 'L1'
    target_stereoa_present = prediction['target'] == 'STEREO-A'
    target_stereob_present = prediction['target'] == 'STEREO-B'
    target_solo_present = prediction['target'] == 'Solar Orbiter'
    target_bepi_present = prediction['target'] == 'BepiColombo'
    target_psp_present = prediction['target'] == 'Parker Solar Probe'
    target_mes_present = prediction['target'] == 'MESSENGER'
    target_vex_present = prediction['target'] == 'Venus Express'

    target_mercury_present = prediction['target'] == 'Mercury'
    target_venus_present = prediction['target'] == 'Venus'
    target_earth_present = prediction['target'] == 'Earth'
    target_mars_present = prediction['target'] == 'Mars'
    target_jupiter_present = prediction['target'] == 'Jupiter'
    target_saturn_present = prediction['target'] == 'Saturn'
    target_uranus_present = prediction['target'] == 'Uranus'
    target_neptune_present = prediction['target'] == 'Neptune'

    if target_l1_present.any():
        dt_L1, dv_L1 = assess_prediction(prediction, 'L1', L1_istime, L1_isspeed)

    if target_stereoa_present.any():
        dt_stereoa, dv_stereoa = assess_prediction(prediction, 'STEREO-A', STEREOA_istime, STEREOA_isspeed)

    if target_stereob_present.any():
        dt_stereob, dv_stereob = assess_prediction(prediction, 'STEREO-B', STEREOB_istime, STEREOB_isspeed)
        
    if target_solo_present.any():
        dt_solo, dv_solo = assess_prediction(prediction, 'Solar Orbiter', SOLO_istime, SOLO_isspeed)
        
    if target_bepi_present.any():
        dt_bepi, dv_bepi = assess_prediction(prediction, 'BepiColombo', BEPI_istime, BEPI_isspeed)
        
    if target_psp_present.any():
        dt_psp, dv_psp = assess_prediction(prediction, 'Parker Solar Probe', PSP_istime, PSP_isspeed)
        
    if target_mes_present.any():
        dt_mes, dv_mes = assess_prediction(prediction, 'MESSENGER', MES_istime, MES_isspeed)

    if target_vex_present.any():
        dt_vex, dv_vex = assess_prediction(prediction, 'Venus Express', VEX_istime, VEX_isspeed)
        
    if target_mercury_present.any():
        dt_mercury, dv_mercury = assess_prediction(prediction, 'Mercury', MERCURY_istime, MERCURY_isspeed)
        
    if target_venus_present.any():
        dt_venus, dv_venus = assess_prediction(prediction, 'Venus', VENUS_istime, VENUS_isspeed)
        
    if target_earth_present.any():
        dt_earth, dv_earth = assess_prediction(prediction, 'Earth', EARTH_istime, EARTH_isspeed)

    if target_mars_present.any():
        dt_mars, dv_mars = assess_prediction(prediction, 'Mars', MARS_istime, MARS_isspeed)
        
    if target_jupiter_present.any():
        dt_jupiter, dv_jupiter = assess_prediction(prediction, 'Jupiter', JUPITER_istime, JUPITER_isspeed)
        
    if target_saturn_present.any():
        dt_saturn, dv_saturn = assess_prediction(prediction, 'Saturn', SATURN_istime, SATURN_isspeed)
        
    if target_uranus_present.any():
        dt_uranus, dv_uranus = assess_prediction(prediction, 'Uranus', URANUS_istime, URANUS_isspeed)

    if target_neptune_present.any():
        dt_neptune, dv_neptune = assess_prediction(prediction, 'Neptune', NEPTUNE_istime, NEPTUNE_isspeed)
    
if __name__ == '__main__':
    main()
