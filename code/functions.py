import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from datetime import datetime
import matplotlib.dates as mdates
from astropy.time import Time as atime
import matplotlib.patheffects as path_effects
from astropy import constants as const
from sunpy.coordinates import frames, get_horizons_coord
import csv
import pandas as pd
import seaborn as sns
import json
import pickle
import logging
import sys
import pdb
import gc

# Constants
AU = const.au.to_value('km')
rsun = const.R_sun.to_value('km')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def calculate_new_time_axis(start_time, end_time, num_elements):
    # Function to calculate new time axis
    time_step = (end_time - start_time) / (num_elements - 1)
    return pd.date_range(start=start_time, end=end_time, freq=time_step)

def merge_tracks(event_path, prediction_path):
    
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)
    
    # Set Seaborn's color palette to a colorblind-friendly palette
    sns.set_palette("colorblind")

    # Create an array to store mean values
    mean_values = np.zeros(30)
    std_values = np.zeros(30)
    
    plt.figure(figsize=(10, 6), facecolor='white')

    # List all files in the folder with .csv ending
    files = [file for file in os.listdir(event_path) if file.endswith('.csv')]
    
    interpolated_tracks = np.zeros((len(files),30))

    # Calculate common new time axis
    all_time_data = []
    for file in files:
        df = pd.read_csv(event_path + file)
        all_time_data.extend(pd.to_datetime(df['TRACK_DATE']))

    min_time = min(all_time_data)
    max_time = max(all_time_data)
    new_time_axis = calculate_new_time_axis(min_time, max_time, num_elements=30)

    # Get a list of distinguishable colors from the colorblind palette
    num_colors = len(files)
    colors = sns.color_palette("colorblind", n_colors=num_colors)

    # Loop through each file and plot data
    for track_num, (file, color) in enumerate(zip(files, colors), start=1):
        df = pd.read_csv(event_path + file)

        # Convert TRACK_DATE to datetime
        df['TRACK_DATE'] = pd.to_datetime(df['TRACK_DATE'])

        # original data
        plt.plot(df['TRACK_DATE'], df['ELON'], 'x', label=f'Track {track_num}', color=color)

        # Convert datetime values to Unix timestamps in seconds
        x = (df['TRACK_DATE'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

        # Create an interpolation function using interp1d
        interp_func = interp1d(x, df['ELON'], kind='linear', fill_value="extrapolate")

        # Convert new_time_axis to Unix timestamps in seconds
        x_new = (new_time_axis - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

        # Interpolate elongation onto the common new time axis
        interp_values = interp_func(x_new)

        # Plot interpolated data as points with matching color
        plt.plot(new_time_axis, interp_values, 'o', label='Interpolated Data', color=color)

        interpolated_tracks[track_num -1,:] = interp_values


    # Calculate the mean values for each element across tracks
    mean_values = np.mean(interpolated_tracks, axis=0)

    # Calculate the standard deviations for each element across tracks
    std_values = np.std(interpolated_tracks, axis=0)

    # Calculate mean values by dividing by the number of files
    num_files = len(files)

    # Plot the mean values as a line
    plt.plot(new_time_axis, mean_values, '-', label='Mean Elongation', markersize=8)

    # Plot the standard deviations as error bars
    #plt.errorbar(new_time_axis, mean_values, yerr=std_values, fmt='o', label='Standard Deviation')

    # Customize the plot
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Elongation', fontsize=14)
    plt.title('Interpolated Elongation Data on Common Time Axis', fontsize=16)
    plt.legend()
    plt.grid(True)

    # Format x-axis tick labels with year
    date_format = mdates.DateFormatter('%Y-%m-%d \n %H:%M:%S')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))

    plt.tight_layout()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Show the plot
    #plt.show()
    
    plt.savefig(prediction_path+'tracks.png', dpi=300, bbox_inches='tight')
    
    plt.clf()
    plt.close()
                
    # Create a DataFrame with renamed columns
    track = pd.DataFrame({'time': new_time_axis, 'elongation': mean_values, 'std': std_values})

    if not os.path.exists(event_path+'interpolated_track'):
        os.mkdir(event_path+'interpolated_track')

    # Save the DataFrame to a CSV file
    track.to_csv(event_path+'interpolated_track/mean_track.csv', index=False)
        
    track.to_csv(prediction_path+'mean_track.csv', index=False)
    
    return track

def fpf_function(xdata, phi, speed):
    # function to fit time-elongation profile of CME
    # assumptions: constant speed, fixed direction
    elon_fit = np.arctan((speed * xdata * np.sin(phi)) / (1. - speed * xdata * np.cos(phi)))

    return elon_fit

def fpf(track, startcut, endcut, prediction_path):
    
    #df = pd.read_csv(track_data)
    df = track
    elon = df["elongation"][startcut:endcut]
    std = df["std"][startcut:endcut]

    # Convert the "time" column to datetime objects
    df["time"] = pd.to_datetime(df["time"])

    time_nu = df["time"][startcut:endcut]

    # Convert the 'time' column to a numerical time array
    time_n = time_nu.apply(lambda x: x.timestamp())

    # calculate launch time of CME
    launch_elon = 0

    # Create an interpolating function to extrapolate time_num based on elon down to `launch_elon`
    f = interp1d(elon, time_n, fill_value="extrapolate")

    # Extrapolate time_num up to `desired_elon`
    launch_time_n = f(launch_elon)
    launch_time = np.datetime64('1970-01-01') + np.array(launch_time_n, dtype='timedelta64[s]')

    time = df["time"][startcut:endcut]

    #fixed_timestamp = time.iloc[0] - pd.DateOffset(launch_time_num)

    # Subtract the fixed timestamp from each element in the 'time' array
    time_num = (time - launch_time).dt.total_seconds()

    xdata = time_num
    ydata = elon
    
    
    # Initial guess for parameters phi and speed
    phi_init = np.deg2rad(60.)
    speed_init = 500./AU 

    bounds = ([0, 200./AU], [np.pi, 4000./AU])  # Example bounds: phi between 0 and pi/2, speed >= 0

    ydata_rad = np.deg2rad(ydata)

    # Perform the curve fitting
    params, covariance = curve_fit(fpf_function, xdata, ydata_rad, p0=[phi_init, speed_init], bounds=bounds)

    # Extract the fitted parameters
    phi_fit, speed_fit = params

    # Generate the fitted curve using the fitted parameters
    elon_fit = fpf_function(xdata, phi_fit, speed_fit)

    # Create the plot and get the axes object
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')  # Adjust figure size as needed

    # Plot the data and fitted curve
    ax.scatter(time, ydata, label="Data")
    ax.plot(time, np.rad2deg(elon_fit), label="Fixed-Phi Fit")

    # Set the x-axis label
    ax.set_xlabel("Time (t)", fontsize=14)

    # Set the y-axis label
    ax.set_ylabel("Elongation [°]", fontsize=14)

    # Set the title
    ax.set_title("Fixed-Phi Fit", fontsize=16)

    # Format the datetime axis labels
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Customize the x-axis ticks for 12-hour intervals
    hour_locator = mdates.HourLocator(interval=12)
    ax.xaxis.set_major_locator(hour_locator)

    ax.legend(fontsize=14)

    p_rounded = round(np.rad2deg(phi_fit))
    s_rounded = round(speed_fit * AU)

    results_text = 'Fitting results:'
    phi_text = f'$\\phi$ = {p_rounded}°'
    speed_text = f'V$_{{const.}}$ = {s_rounded} km s$^{{-1}}$'

    underlined_effect = [path_effects.SimpleLineShadow(),
                        path_effects.Normal()]
    text_loc = round(len(time)*0.7)

    # Coordinates for placing the text annotations
    results_x, results_y = time[text_loc], round(max(ydata) - min(ydata)) * 0.5 + round(min(ydata))
    speed_x, speed_y = time[text_loc], round(max(ydata) - min(ydata)) * 0.5 + round(min(ydata)) - 0.5
    phi_x, phi_y = time[text_loc], round(max(ydata) - min(ydata)) * 0.5 + round(min(ydata)) - 1

    # Text annotations to the plot
    ax.text(results_x, results_y, results_text, fontsize=16, path_effects=underlined_effect)
    ax.text(phi_x, phi_y, phi_text, fontsize=14)
    ax.text(speed_x, speed_y, speed_text, fontsize=14)

    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)

    plt.savefig(prediction_path+'FPF_.png', dpi=150, bbox_inches='tight')
    
    plt.clf()
    plt.close()
    
    # Arrival time at 1 AU

    travel_time = AU/s_rounded

    FPF_Earth_arrival_np = np.datetime64('1970-01-01') + np.array((launch_time_n + travel_time), dtype='timedelta64[s]')
    # Convert FPF_Earth_arrival_np to a regular datetime object
    FPF_Earth_arrival = FPF_Earth_arrival_np.astype(datetime)
    formatted_arrival = FPF_Earth_arrival.strftime("%Y-%m-%dT%H:%M")
    launch = launch_time.astype(datetime)
    formatted_launch = launch.strftime("%Y-%m-%dT%H:%M")

    # Print the fitted parameters
    print('    ')
    print('-------Fixed-Phi Fitting Results:-------')
    print('----------------------------------------')
    print("|| Launch time:", formatted_launch)
    print("|| Fitted phi:", round(np.rad2deg(phi_fit)), 'degrees from observer')
    print("|| Fitted speed:", round(speed_fit*AU), 'km/s')
    print("|| 1 AU arrival:", formatted_arrival)
    print("|| Maximum elongation used: ", round(max(ydata),1), 'degrees')
    print('----------------------------------------')
    print('    ')
    
    # save results 

    fpf_results = {
        'launch_time_FPF': launch_time,
        'phi_FPF': round(np.rad2deg(phi_fit)),
        'speed_FPF': round(speed_fit * AU),
        'arrivaltime_L1_FPF': FPF_Earth_arrival
    }

    with open(prediction_path + 'FPF_results.pkl', 'wb') as file:
        pickle.dump(fpf_results, file)
    
    return fpf_results

def ELCon(elon, d, phi, hwidth, f):
    """ ELlipse Conversion method (Rollett et al., 2016)

    converts HI time-elongation measurements to time-distance.
    """
    p = phi
    l = hwidth
    e = elon
    R_ell = np.zeros(np.size(elon))

    for i in range(len(elon)):
        
        beta1 = np.pi - e[i] - p

        if beta1 > np.pi/2.:
            beta1 = e[i] + p

        theta = np.arctan(f**2 * np.tan(beta1))
        w = np.sqrt((np.cos(theta)**2) * (f**2 - 1) + 1)

        thetas = np.arctan(f**2 * np.tan(l))
        ws = np.sqrt((np.cos(thetas)**2) * (f**2 - 1) + 1)

        X = ((np.cos(l-thetas)/np.sin(l)) + ws)**(-1) * ((np.cos(e[i]+p+theta)/(w*np.sin(e[i]+p))) + 1)

        if (np.pi-e[i]-p) > np.pi/2:
            X = ((np.cos(l-thetas)/np.sin(l))+ws)**(-1) * ((-np.sin(np.pi/2+theta-e[i]-p)/(w*np.sin(e[i]+p))) + 1)

        Y = (d * np.sin(e[i]))/(np.sin(e[i] + p))

        R_ell[i] = Y/(1 - ws*X)
        
    return R_ell

def fitdbm(x, gamma):
    """ function to fit time-distance profile of CME """   
    fit = (1/gamma) * np.log(1 + gamma*(vinit - swspeed) * x) + swspeed*x + rinit
    
    return fit

def fitdbmneg(x, gamma):
    """ function to fit time-distance profile of CME """   
    fit = (-1/gamma) * np.log(1 - gamma*(vinit - swspeed) * x) + swspeed*x + rinit
    
    return fit

def cost_function(gamma):
    predicted = fitdbm(xdata, gamma)
    return np.sum((ydata - predicted) ** 2)

def cost_functionneg(gamma):
    predicted = fitdbmneg(xdata, gamma)
    return np.sum((ydata - predicted) ** 2)

def DBMfitting(time, distance_au, prediction_path, det_plot, startfit = 1, endfit = 20, silent = 1, max_residual = 1.5, max_gamma = 3e-7):
    """ fit the ELCon time-distance track using the drag-based equation of motion from Vrsnak et al. (2013) """
    global tinit, rinit, vinit, swspeed, xdata, ydata

    # start end end points of fitting.
    # This was done manually so far, but should be implemented in a way that it is done automatically.
    # However, it might always be necessary that a forecaster reviews the fitting because not each HI kinematics can
    # be fitted from each startpoint on.

    # startfit = 3 -> default settings
    # endfit = 18 -> default settings
    # Convert datetime values to seconds since the first element
    time_num = (time - time.iloc[0]).dt.total_seconds()

    speedtime = time_num - time_num[0]
    distance_km = distance_au * AU
    distance_rsun = distance_au * AU / rsun

    #build time derivative from ELCon CME time-distance track
    speed = np.gradient(distance_km, speedtime)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #!!! How do we build the speed errors? In IDL this is done by the function DERIVSIG.
    # In Python, I have no idea! -> This needs to be solved.
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    tinit = time[startfit]
    rinit = distance_km[startfit]
    vinit = speed[startfit]

    xdata = speedtime[startfit:endfit]
    ydata = distance_km[startfit:endfit]

    winds = np.arange(200, 775, 25)
    fit = np.zeros((len(winds), len(xdata)))
    fitspeed = np.zeros((len(winds), len(xdata)))
    residuals = np.zeros((len(winds), len(xdata)))
    gamma = np.zeros(len(winds))
    success = np.zeros(len(winds))
    
    # Initial guess for the parameter gamma
    initial_guess = 1e-7

    # do a fit for each wind speed separately and plot valid fits
    for i in range(len(winds)):
        swspeed = winds[i]
        if vinit > swspeed:
            # Perform the optimization
            result = minimize(cost_function, initial_guess, method='Nelder-Mead')
            # Print the fitted parameter
            if silent == 0:
                print('=====') 
                print('wind: ', winds[i], 'success: ', result.success)
            success[i] = result.success
            if result.success:
                gamma_fit = result.x[0]
                if silent == 0:
                    print(f"Fitted gamma: {round(gamma_fit*1e7, 2)} 1e-7 1/km")
                fit_ = fitdbm(xdata, gamma_fit)
                gamma[i] = gamma_fit
                fit[i,:] = fit_   
                residuals[i,:] = ydata - fit_
                if silent == 0:
                    print('mean_res: ', round(np.mean(residuals[i,:])/rsun, 2), 'solar radii')
                fitspeed[i,:] = np.gradient(fit[i,:], xdata)
                if silent == 0:
                    print('-----')            
                    print('positive')
            else:
                continue

        else:
            # Perform the optimization
            result = minimize(cost_functionneg, initial_guess, method='Nelder-Mead')
            if silent == 0:
                print('=====')
                print('wind: ', winds[i], 'success: ', result.success)
            success[i] = result.success
            # Print the fitted parameter
            if result.success:
                gamma_fit = result.x[0]
                if silent == 0:
                    print(f"Fitted gamma: {round(gamma_fit*1e7, 2)} 1e-7 1/km")
                fit_ = fitdbmneg(xdata, gamma_fit)
                gamma[i] = gamma_fit
                fit[i,:] = fit_   
                residuals[i,:] = ydata - fit_
                fitspeed[i,:] = np.gradient(fit[i,:], xdata)
                if silent == 0:
                    print('mean_res: ', round(np.mean(residuals[i,:])/rsun, 2), 'solar radii')
                    print('-----')            
                    print('negative')
            else:
                continue
            
    res = np.zeros(len(winds))

    for i in range(len(winds)):
        if success[i]:
            res[i] = np.mean(residuals[i,:])
        else:
            res[i] = np.nan
    
    #pdb.set_trace()
            
    if det_plot:
        fig, ax = plt.subplots(1, 1, figsize = (12,4), dpi = 300, facecolor='white')

        ax.set_title('ELEvoHI DBMfits', size = 20)
        ax.set_ylabel('CME Apex Speed [km s$^{-1}$]', fontsize = 20.0)
        ax.set_xlabel('Heliocentric Distance [R$_\odot$]', fontsize = 20)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 15)

        norm = mpl.colors.Normalize(vmin = winds.min(), vmax = winds.max())
        cmap = mpl.cm.ScalarMappable(norm = norm, cmap = mpl.cm.jet)
        cmap.set_array([])

        ax = plt.gca()
        cbar = plt.colorbar(cmap, pad = 0.02, ax=ax)
        cbar.set_label('solar wind speed [km s$^{-1}$]', rotation = 270, fontsize = 15, labelpad = 20)
        cbar.ax.tick_params(axis='y', direction='inout', length=5, width=0, labelsize=12)
    
    gamma_v = []
    res_v = []
    winds_v = []
    
    print('')
    print(f"Fits with residuals smaller than {max_residual} solar radii are used.")
    print(f"The drag parameter is limited to be a maximum of {max_gamma} /km.")
    print('')

    for i in range(len(winds)):
        if success[i] != True:
            continue
        if np.abs(res[i]/rsun) > max_residual:
            continue
        if gamma[i] < 0:
            continue
        if max_gamma >= np.abs(gamma[i]):
            gamma_v.append(gamma[i])
            res_v.append(res[i]/rsun)
            winds_v.append(winds[i])
            if silent == 0:
                print('------------')
                print('Valid fit:')
                print('i: ', i)
                print('Gamma: ', round(gamma[i]*1e7, 2), '1e-7 1/km')
                print('Mean Residual: ', round(res[i]/rsun, 2), 'solar radii')
                print('Wind speed: ', winds[i])
            if det_plot:
                ax.plot(distance_rsun[startfit:endfit], fitspeed[i,:], '-', label='fit', c=cmap.to_rgba(winds[i]))
        else:
            if silent == 0:
                print('------------')
                print('i: ',i)
                print('wrong gamma: ', round(gamma[i]*1e7, 2), '1e-7 1/km')
                print(round(res[i]/rsun, 2))
            continue

    if det_plot:
        ax.plot(distance_rsun, speed, 'o', label='data', c='black')
        
        # Calculate y-axis limits
        # y_lower = min(speed) - 100
        # y_upper = max(speed) + 100
        
        # Use fixed y-axis limits
        y_lower = 0
        y_upper = 1200

        # Set the y-axis limits
        ax.set_ylim(y_lower, y_upper)
    
    # find the most accurate fitting parameters based on the minimum mean residual
    
    gamma_valid = np.array(gamma_v)   
    res_valid = np.array(res_v)   
    winds_valid = np.array(winds_v)   
    min_res = np.argmin(np.abs(res_valid))
       
    print('')
    print('===========================================================')
    print('|| Most accurate fitting for solar wind speed: ' + np.array2string(winds_valid[min_res]) + ' [km/s].||')
    print('|| Resulting drag paramter: ' + np.array2string(gamma_valid[min_res]) + ' [1/km].        ||')
    print('|| Minimum mean residual: ' + np.array2string(round(res_valid[min_res], 2)) + ' [Rsun].                    ||')
    print('|| Initial time: ' + tinit.strftime('%Y-%m-%d %H:%M') + ' UT                      ||')
    print('|| Initial distance: ' + np.array2string(round(rinit/rsun, 1)) + ' [Rsun].                         ||')   
    print('|| Initial speed: ' + str(round(vinit)) + ' [km/s].                             ||')  
    print('===========================================================')
    print('')

    # Get the indices that would sort the array based on absolute values
    sorted_indices = np.argsort(np.abs(res_valid))
    gamma_valid = gamma_valid[sorted_indices]
    res_valid = res_valid[sorted_indices]
    winds_valid = winds_valid[sorted_indices]
    
    #pdb.set_trace()
    
    if det_plot:
        filename = prediction_path + '/DBMfit.png' 
        plt.savefig(filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')       
        fig.clf()
        plt.close(fig)
    
    return gamma_valid, winds_valid, res_valid, tinit, rinit, vinit, swspeed, xdata, ydata

def elevo_analytic(R, f, halfwidth, delta, out=False, plot=False):
    """
    Calculate the distance from the Sun to a point along an ellipse in the ecliptic plane, which 
    describes the front of a solar transient. This point is specified by the angle delta 
    between the ellipse apex and the in situ spacecraft. The ellipse is specified by 
    the distance of the apex (R), the half width in degrees in heliospheric longitude, and the aspect ratio.
    One of the ellipse main axes is oriented along the propagation direction, the 
    other perpendicular to it. The aspect ratio can take any values, but physical 
    ones suitable for CMEs cluster most likely around 1.3 +/ 0.2.

    Authors: Christian Möstl, Tanja Amerstorfer, Jürgen Hinterreiter, Maike Bauer
    
    Parameters: 
        R (numpy.ndarray): Heliocentric distance of CME apex in AU
        aspectratio (float): Ellipse aspect ratio (a/b)
        halfwidth (float): Halfwith of the ellipse
        delta (float): Angular separation of CME apex and in situ s/c

    Returns:
        numpy.ndarray: Heliocentric distance of the CME front along the Sun-s/c line.

    History:
        2014/09: Version 1.0 numerical solution (C. Möstl)
        2014/10: Replaced numerical with analytic procedure (C. Möstl)
        2015/11 - 2019/08: several changes related to ELEvoHI (T. Amerstorfer, J. Hinterreiter)
        2023/08: translated from IDL to Python (T. Amerstorfer)

    """
    
    aspectratio = 1/f

    if abs(delta) >= halfwidth:
        # Half width lambda must be greater than delta!
        n = R * np.nan
        print("No hit predicted for this target.")
        return n

    # Construct ellipse
    theta = np.arctan(f**2 * np.tan(halfwidth))
    #print('theta: ', theta)
    omega = np.sqrt(np.cos(theta)**2 * (f**2 - 1) + 1)
    b = R * omega * np.sin(halfwidth) / (np.cos(halfwidth - theta) + omega * np.sin(halfwidth))
    a = b / f
    c = R - b

    # Ellipse parameters necessary for plotting
    if out:
        print('a: ', a)
        print('b: ', b)
        print('c: ', c)
        print('R: ', R)
        print('f: ', f)
        print('distance of apex in AU: ', R)
    
    if plot:
        return a, b, c
    else:
        # Distance from Sun of given point along ellipse front
        root = np.sqrt(np.sin(delta)**2 * f**2 * (b**2 - c**2) + np.cos(delta)**2 * b**2)
        dvalue_analytic_front = (c * np.cos(delta) + root) / (np.sin(delta)**2 * f**2 + np.cos(delta)**2)
        dvalue_analytic_rear = (c * np.cos(delta) - root) / (np.sin(delta)**2 * f**2 + np.cos(delta)**2)
        dvalue = dvalue_analytic_front
        if out:
            print('distance d in AU: ', dvalue)
        return dvalue
   
def elevo(R, time_array, tnum, direction, f, halfwidth, vdrag, track, availability, hit_counts, delta_values, positions, HIobs, outer_system, prediction_path, det_plot, movie=False, timegrid = 1440):
    # calculate ELEvo radial distances in direction of each target and
    # arrival times and speeds
    
    # Assign values from dictionary to variables
    sta_available = availability['sta_available']
    stb_available = availability['stb_available']
    mes_available = availability['mes_available']
    vex_available = availability['vex_available']
    solo_available = availability['solo_available']
    psp_available = availability['psp_available']
    bepi_available = availability['bepi_available']
    
    hit_sta = hit_counts['hit_sta']
    hit_stb = hit_counts['hit_stb']
    hit_mes = hit_counts['hit_mes']    
    hit_vex = hit_counts['hit_vex']
    hit_solo = hit_counts['hit_solo']
    hit_psp = hit_counts['hit_psp']
    hit_bepi = hit_counts['hit_bepi']
    hit_L1 = hit_counts['hit_L1']
    hit_mercury = hit_counts['hit_mercury']
    hit_venus = hit_counts['hit_venus']
    hit_earth = hit_counts['hit_earth']
    hit_mars = hit_counts['hit_mars']
    
    if outer_system:
        hit_jupiter = hit_counts['hit_jupiter']
        hit_saturn = hit_counts['hit_saturn']
        hit_uranus = hit_counts['hit_uranus']
        hit_neptune = hit_counts['hit_neptune']
        delta_jupiter = delta_values['delta_jupiter']
        delta_saturn = delta_values['delta_saturn']
        delta_uranus = delta_values['delta_uranus']
        delta_neptune = delta_values['delta_neptune']
    
    if sta_available:
        delta_sta = delta_values['delta_sta']
    if stb_available:
        delta_stb = delta_values['delta_stb']
    if mes_available:
        delta_mes = delta_values['delta_mes']
    if vex_available:
        delta_vex = delta_values['delta_vex']
    if psp_available:
        delta_psp = delta_values['delta_psp']
    if solo_available:
        delta_solo = delta_values['delta_solo']
    if bepi_available:
        delta_bepi = delta_values['delta_bepi']
        
    delta_L1 = delta_values['delta_L1']
    delta_mercury = delta_values['delta_mercury']
    delta_venus = delta_values['delta_venus']
    delta_earth = delta_values['delta_earth']
    delta_mars = delta_values['delta_mars']
      
    mars_r = positions['mars_r']
    earth_r = positions['earth_r']
    venus_r = positions['venus_r']
    mercury_r = positions['mercury_r']
    L1_r = positions['L1_r']
    mars_lon = positions['mars_lon']
    earth_lon = positions['earth_lon']
    venus_lon = positions['venus_lon']
    mercury_lon = positions['mercury_lon']
    L1_lon = positions['L1_lon']
    
    if sta_available:
        sta_r = positions['sta_r']
        sta_lon = positions['sta_lon']        
    if stb_available:
        stb_r = positions['stb_r']
        stb_lon = positions['stb_lon']    
    if bepi_available:   
        bepi_r = positions['bepi_r']
        bepi_lon = positions['bepi_lon']    
    if solo_available:
        solo_r = positions['solo_r']
        solo_lon = positions['solo_lon']
    if psp_available:
        psp_r = positions['psp_r']
        psp_lon = positions['psp_lon']
    if vex_available:
        vex_r = positions['vex_r']
        vex_lon = positions['vex_lon']
    if mes_available:
        mes_r = positions['mes_r']
        mes_lon = positions['mes_lon']
    if outer_system:
        neptune_r = positions['neptune_r']
        uranus_r = positions['uranus_r']
        saturn_r = positions['saturn_r']
        jupiter_r = positions['jupiter_r']
        neptune_lon = positions['neptune_lon']
        uranus_lon = positions['uranus_lon']
        saturn_lon = positions['saturn_lon']
        jupiter_lon = positions['jupiter_lon']
          
    pred = []
    
    # Calculate time differences needed for deriving the speeds
    time_diff = np.diff(tnum)
    
    hit = 0
    

    if sta_available == 1 and hit_sta == 1:
        ############# STEREO-A #############
        d_sta = elevo_analytic(R, f, halfwidth, delta_sta, out=False)

        # Find the index of the closest value
        index_sta = np.argmin(np.abs(d_sta - sta_r))

        # arrival time at STEREO-A
        arrtime_sta = time_array[index_sta]

        # arrival speed at STEREO-A
        distance_diff = np.diff(d_sta*AU)
        speed_sta = distance_diff / time_diff
        arrspeed_sta = speed_sta[index_sta-1] # first element is cut off during building the time derivative

        print("------------------------------------")
        print("Arrival time at STEREO-A:", arrtime_sta.strftime("%Y-%m-%d %H:%M"))
        print(f"Arrival speed at STEREO-A: {arrspeed_sta:.0f} km/s")
        
        hit = 1
        
        pred.append({"target": "STEREO-A", "arrival time [UT]": arrtime_sta.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_sta)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if stb_available == 1 and hit_stb == 1:
        ############# STEREO-B #############
        d_stb = elevo_analytic(R, f, halfwidth, delta_stb, out=False)

        # Find the index of the closest value
        index_stb = np.argmin(np.abs(d_stb - stb_r))

        # arrival time at STEREO-B
        arrtime_stb = time_array[index_stb]

        # arrival speed at STEREO-B
        distance_diff = np.diff(d_stb*AU)
        speed_stb = distance_diff / time_diff
        arrspeed_stb = speed_stb[index_stb-1] # first element is cut off during building the time derivative

        print("------------------------------------")
        print("Arrival time at STEREO-B:", arrtime_stb.strftime("%Y-%m-%d %H:%M"))
        print(f"Arrival speed at STEREO-B: {arrspeed_stb:.0f} km/s")
        
        hit = 1

        pred.append({"target": "STEREO-B", "arrival time [UT]": arrtime_stb.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_stb)), "dt [h]": np.nan, "dv [km/s]": np.nan})        
        
    if mes_available == 1 and hit_mes == 1:
        ############# MESSENGER #############
        d_mes = elevo_analytic(R, f, halfwidth, delta_mes, out=False)

        # Find the index of the closest value
        index_mes = np.argmin(np.abs(d_mes - mes_r))

        # arrival time at MESSENGER
        arrtime_mes = time_array[index_mes]

        # arrival speed at MESSENGER
        distance_diff = np.diff(d_mes*AU)
        speed_mes = distance_diff / time_diff
        arrspeed_mes = speed_mes[index_mes-1] # first element is cut off during building the time derivative

        print("------------------------------------")
        print("Arrival time at MESSENGER:", arrtime_mes.strftime("%Y-%m-%d %H:%M"))
        print(f"Arrival speed at MESSENGER: {arrspeed_mes:.0f} km/s")
        
        hit = 1

        pred.append({"target": "MESSENGER", "arrival time [UT]": arrtime_mes.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_mes)), "dt [h]": np.nan, "dv [km/s]": np.nan})
        
    if vex_available == 1 and hit_vex == 1:
        ############# Venus Express #############
        d_vex = elevo_analytic(R, f, halfwidth, delta_vex, out=False)

        # Find the index of the closest value
        index_vex = np.argmin(np.abs(d_vex - vex_r))

        # arrival time at Venus Express
        arrtime_vex = time_array[index_vex]

        # arrival speed at Venus Express
        distance_diff = np.diff(d_vex*AU)
        speed_vex = distance_diff / time_diff
        arrspeed_vex = speed_vex[index_vex-1] # first element is cut off during building the time derivative

        print("------------------------------------")
        print("Arrival time at Venus Express:", arrtime_vex.strftime("%Y-%m-%d %H:%M"))
        print(f"Arrival speed at Venus Express: {arrspeed_vex:.0f} km/s")
        
        hit = 1
        
        pred.append({"target": "Venus Express", "arrival time [UT]": arrtime_vex.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_vex)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if solo_available == 1 and hit_solo == 1:
        ############# Solar Orbiter #############
        d_solo = elevo_analytic(R, f, halfwidth, delta_solo, out=False)

        # Find the index of the closest value
        index_solo = np.argmin(np.abs(d_solo - solo_r))

        # arrival time at Solar Orbiter
        arrtime_solo = time_array[index_solo]

        # arrival speed at Solar Orbiter
        distance_diff = np.diff(d_solo*AU)
        speed_solo = distance_diff / time_diff
        arrspeed_solo = speed_solo[index_solo-1] # first element is cut off during building the time derivative

        print("------------------------------------")
        print("Arrival time at Solar Orbiter:", arrtime_solo.strftime("%Y-%m-%d %H:%M"))
        print(f"Arrival speed at Solar Orbiter: {arrspeed_solo:.0f} km/s")
        
        hit = 1
        
        pred.append({"target": "Solar Orbiter", "arrival time [UT]": arrtime_solo.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_solo)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if psp_available == 1 and hit_psp == 1:
        ############# Parker Solar Probe #############
        d_psp = elevo_analytic(R, f, halfwidth, delta_psp, out=False)

        # Find the index of the closest value
        index_psp = np.argmin(np.abs(d_psp - psp_r))

        # arrival time at Parker Solar Probe
        arrtime_psp = time_array[index_psp]

        # arrival speed at Parker Solar Probe
        distance_diff = np.diff(d_psp*AU)
        speed_psp = distance_diff / time_diff
        arrspeed_psp = speed_psp[index_psp-1] # first element is cut off during building the time derivative

        print("------------------------------------")
        print("Arrival time at Parker Solar Probe:", arrtime_psp.strftime("%Y-%m-%d %H:%M"))
        print(f"Arrival speed at Parker Solar Probe: {arrspeed_psp:.0f} km/s")
        
        hit = 1
        
        pred.append({"target": "Parker Solar Probe", "arrival time [UT]": arrtime_psp.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_psp)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if bepi_available == 1 and hit_bepi == 1:
        ############# BepiColombo #############
        d_bepi = elevo_analytic(R, f, halfwidth, delta_bepi, out=False)

        # Find the index of the closest value
        index_bepi = np.argmin(np.abs(d_bepi - bepi_r))

        # arrival time at BepiColombo
        arrtime_bepi = time_array[index_bepi]

        # arrival speed at BepiColombo
        distance_diff = np.diff(d_bepi*AU)
        speed_bepi = distance_diff / time_diff
        arrspeed_bepi = speed_bepi[index_bepi-1] # first element is cut off during building the time derivative

        print("------------------------------------")
        print("Arrival time at BepiColombo:", arrtime_bepi.strftime("%Y-%m-%d %H:%M"))
        print(f"Arrival speed at BepiColombo: {arrspeed_bepi:.0f} km/s")
        
        hit = 1
        
        pred.append({"target": "BepiColombo", "arrival time [UT]": arrtime_bepi.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_bepi)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if hit_mercury == 1:
        ############# Mercury #############
        d_mercury = elevo_analytic(R, f, halfwidth, delta_mercury, out=False)

        # Find the index of the closest value
        index_mercury = np.argmin(np.abs(d_mercury - mercury_r))

        # arrival time at Mercury
        arrtime_mercury = time_array[index_mercury]

        # arrival speed at Mercury
        distance_diff = np.diff(d_mercury*AU)
        speed_mercury = distance_diff / time_diff
        arrspeed_mercury = speed_mercury[index_mercury-1] # first element is cut off during building the time derivative

        print("------------------------------------")
        print("Arrival time at Mercury:", arrtime_mercury.strftime("%Y-%m-%d %H:%M"))
        print(f"Arrival speed at Mercury: {arrspeed_mercury:.0f} km/s")
        
        hit = 1
        
        pred.append({"target": "Mercury", "arrival time [UT]": arrtime_mercury.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_mercury)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if hit_venus == 1:
        ############# Venus #############
        d_venus = elevo_analytic(R, f, halfwidth, delta_venus, out=False)

        # Find the index of the closest value
        index_venus = np.argmin(np.abs(d_venus - venus_r))

        # arrival time at Venus
        arrtime_venus = time_array[index_venus]

        # arrival speed at Venus
        distance_diff = np.diff(d_venus*AU)
        speed_venus = distance_diff / time_diff
        arrspeed_venus = speed_venus[index_venus-1] # first element is cut off during building the time derivative

        print("------------------------------------")
        print("Arrival time at Venus:", arrtime_venus.strftime("%Y-%m-%d %H:%M"))
        print(f"Arrival speed at Venus: {arrspeed_venus:.0f} km/s")
        
        hit = 1
        
        pred.append({"target": "Venus", "arrival time [UT]": arrtime_venus.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_venus)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if hit_L1 == 1:
        ############# L1 #############
        d_L1 = elevo_analytic(R, f, halfwidth, delta_L1, out=False)

        # Find the index of the closest value
        index_L1 = np.argmin(np.abs(d_L1 - L1_r))

        # arrival time at L1
        arrtime_L1 = time_array[index_L1]

        # arrival speed at L1
        distance_diff = np.diff(d_L1*AU)
        speed_L1 = distance_diff / time_diff
        arrspeed_L1 = speed_L1[index_L1-1] # first element is cut off during building the time derivative

        print("------------------------------------")
        print("Arrival time at L1:", arrtime_L1.strftime("%Y-%m-%d %H:%M"))
        print(f"Arrival speed at L1: {arrspeed_L1:.0f} km/s")
        
        hit = 1
        
        pred.append({"target": "L1", "arrival time [UT]": arrtime_L1.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_L1)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if hit_mars == 1:
        ############# Mars #############
        d_mars = elevo_analytic(R, f, halfwidth, delta_mars, out=False)

        # Find the index of the closest value
        index_mars = np.argmin(np.abs(d_mars - mars_r))

        # arrival time at Mars
        arrtime_mars = time_array[index_mars]

        # arrival speed at Mars
        distance_diff = np.diff(d_mars*AU)
        speed_mars = distance_diff / time_diff
        arrspeed_mars = speed_mars[index_mars-1] # first element is cut off during building the time derivative

        print("------------------------------------")
        print("Arrival time at Mars:", arrtime_mars.strftime("%Y-%m-%d %H:%M"))
        print(f"Arrival speed at Mars: {arrspeed_mars:.0f} km/s")
        
        hit = 1
        
        pred.append({"target": "Mars", "arrival time [UT]": arrtime_mars.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_mars)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if outer_system == 1:

        if hit_jupiter == 1:
            ############# Jupiter #############
            d_jupiter = elevo_analytic(R, f, halfwidth, delta_jupiter, out=False)

            # Find the index of the closest value
            index_jupiter = np.argmin(np.abs(d_jupiter - jupiter_r))

            # arrival time at Jupiter
            arrtime_jupiter = time_array[index_jupiter]

            # arrival speed at Jupiter
            distance_diff = np.diff(d_jupiter*AU)
            speed_jupiter = distance_diff / time_diff
            arrspeed_jupiter = speed_jupiter[index_jupiter-1] # first element is cut off during building the time derivative

            print("------------------------------------")
            print("Arrival time at Jupiter:", arrtime_jupiter.strftime("%Y-%m-%d %H:%M"))
            print(f"Arrival speed at Jupiter: {arrspeed_jupiter:.0f} km/s")
            
            hit = 1
            
            pred.append({"target": "Jupiter", "arrival time [UT]": arrtime_jupiter.strftime("%Y-%m-%d %H:%M"),
                         "arrival speed [km/s]": int(round(arrspeed_jupiter)), "dt [h]": np.nan, "dv [km/s]": np.nan})

        if hit_saturn == 1:
            ############# Saturn #############
            d_saturn = elevo_analytic(R, f, halfwidth, delta_saturn, out=False)

            # Find the index of the closest value
            index_saturn = np.argmin(np.abs(d_saturn - saturn_r))

            # arrival time at Saturn
            arrtime_saturn = time_array[index_saturn]

            # arrival speed at Saturn
            distance_diff = np.diff(d_saturn*AU)
            speed_saturn = distance_diff / time_diff
            arrspeed_saturn = speed_saturn[index_saturn-1] # first element is cut off during building the time derivative

            print("------------------------------------")
            print("Arrival time at Saturn:", arrtime_saturn.strftime("%Y-%m-%d %H:%M"))
            print(f"Arrival speed at Saturn: {arrspeed_saturn:.0f} km/s")
            
            hit = 1
            
            pred.append({"target": "Saturn", "arrival time [UT]": arrtime_saturn.strftime("%Y-%m-%d %H:%M"),
                         "arrival speed [km/s]": int(round(arrspeed_saturn)), "dt [h]": np.nan, "dv [km/s]": np.nan})

        if hit_uranus == 1:
            ############# Uranus #############
            d_uranus = elevo_analytic(R, f, halfwidth, delta_uranus, out=False)

            # Find the index of the closest value
            index_uranus = np.argmin(np.abs(d_uranus - uranus_r))

            # arrival time at Uranus
            arrtime_uranus = time_array[index_uranus]

            # arrival speed at Uranus
            distance_diff = np.diff(d_mars*AU)
            speed_uranus = distance_diff / time_diff
            arrspeed_uranus = speed_uranus[index_uranus-1] # first element is cut off during building the time derivative

            print("------------------------------------")
            print("Arrival time at Uranus:", arrtime_uranus.strftime("%Y-%m-%d %H:%M"))
            print(f"Arrival speed at Uranus: {arrspeed_uranus:.0f} km/s")
            
            hit = 1
            
            pred.append({"target": "Uranus", "arrival time [UT]": arrtime_uranus.strftime("%Y-%m-%d %H:%M"),
                         "arrival speed [km/s]": int(round(arrspeed_uranus)), "dt [h]": np.nan, "dv [km/s]": np.nan})

        if hit_neptune == 1:
            ############# Neptune #############
            d_neptune = elevo_analytic(R, f, halfwidth, delta_neptune, out=False)

            # Find the index of the closest value
            index_neptune = np.argmin(np.abs(d_neptune - neptune_r))

            # arrival time at Neptune
            arrtime_neptune = time_array[index_neptune]

            # arrival speed at Neptune
            distance_diff = np.diff(d_neptune*AU)
            speed_neptune = distance_diff / time_diff
            arrspeed_neptune = speed_neptune[index_neptune-1] # first element is cut off during building the time derivative

            print("------------------------------------")
            print("Arrival time at Neptune:", arrtime_neptune.strftime("%Y-%m-%d %H:%M"))
            print(f"Arrival speed at Neptune: {arrspeed_neptune:.0f} km/s")
            
            hit = 1
            
            pred.append({"target": "Neptune", "arrival time [UT]": arrtime_neptune.strftime("%Y-%m-%d %H:%M"),
                         "arrival speed [km/s]": int(round(arrspeed_neptune)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    print("------------------------------------")
    
    if not hit:
        pred.append({"target": np.nan, "arrival time [UT]": np.nan,
                         "arrival speed [km/s]": np.nan, "dt [h]": np.nan, "dv [km/s]": np.nan})
    
    if det_plot:
        # Create a figure with two subplots (panels)
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), facecolor='white')

        # Heliocentric distance
        axs[0].plot(time_array, R, label='CME apex')
        if hit_L1:
            axs[0].plot(time_array, d_L1, label='L1')
        
        axs[0].set_ylabel("Heliocentric Distance [AU]")
        axs[0].set_title("ELEvo kinematics - Heliocentric Distance")
        axs[0].grid(True)
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axs[0].legend()

        # CME speed
        axs[1].plot(time_array[1:], vdrag[1:], label='CME apex')
        if hit_L1:
            axs[1].plot(time_array[1:], speed_L1, label='L1')
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("CME Speed [km s$^{-1}$]")
        axs[1].set_title("ELEvo kinematics - CME Speed")
        axs[1].grid(True)
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Adjust locator for better date spacing
        axs[1].legend()

        # Calculate y-axis limits
        y_lower = min(vdrag) - 100
        y_upper = max(vdrag) + 100

        # Set the y-axis limits
        axs[1].set_ylim(y_lower, y_upper)

        # Rotate x-axis labels for both panels
        for ax in axs:
            ax.xaxis.set_tick_params(rotation=45)

        plt.tight_layout()

        filename = prediction_path + '/ELEvo.png' 

        plt.savefig(filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        
        fig.clf()
        plt.close(fig)
    
    prediction = pd.DataFrame(pred)
    
    # interpolate time-elongation track to movie time-axis to display the tangent to the CME front
    # Convert datetime values to Unix timestamps in seconds
    elon = track["elongation"]

    # existing time axis from tracking
    track["time"] = pd.to_datetime(track["time"])
    x = (track["time"] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    
    # Create an interpolation function using interp1d
    interp_func = interp1d(x, elon, kind='linear', fill_value="extrapolate")

    # Convert new_time_axis to Unix timestamps in seconds
    time_a = pd.to_datetime(time_array)
    
    # Convert datetime values to Unix timestamps in seconds
    x_movie = (time_a - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    
    x_last_element = x.iloc[-1]  # Get the last element of track time

    # Check which elements in x_new are greater than or equal to the last element of x
    smaller_than_last_x = np.where(x_movie <= x_last_element)[0]
    
    x_new = x_movie[smaller_than_last_x]

    # Interpolate elongation onto the common new time axis
    # This new time axis ends with endcut defined in config file
    interp_elon = interp_func(x_new)
    
    # Define the starting point of the tangent
    if HIobs == 'A':
        start_angle = sta_lon
        start_radius = sta_r
    if HIobs == 'B':
        start_angle = stb_lon
        start_radius = stb_r

    # Define the length of the tangent
    tangent_length = 1.2
    
    # Define the length of the HI fov shading
    fov_length = 1
    
    elon_rad = np.deg2rad(interp_elon)
    
    sns.set_context('talk')
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(10, 10))
    backcolor = 'black'
   
    if movie:
        # Initialize your plot outside of the loop
        print('Making frames.')

        if os.path.isdir(prediction_path + '/frames') == False: os.mkdir(prediction_path + '/frames')

        # Loop over time frames
    if movie or det_plot:
        for k in range(timegrid):
            
            if not movie:
                k = 200
            
            t = (np.arange(181) * np.pi/180) - direction
            t1 = (np.arange(181) * np.pi/180)

            a, b, c = elevo_analytic(R[k]*AU, f, halfwidth, 0., out=False, plot=True)
            a = a/AU
            b = b/AU
            c = c/AU

            xc = c * np.cos(direction) + ((a * b) / np.sqrt((b * np.cos(t1)) ** 2 + (a * np.sin(t1)) ** 2)) * np.sin(t)
            yc = c * np.sin(direction) + ((a * b) / np.sqrt((b * np.cos(t1)) ** 2 + (a * np.sin(t1)) ** 2)) * np.cos(t)

            theta_ell = np.arctan2(yc, xc)
            r_ell = np.sqrt(xc ** 2 + yc ** 2)

            if np.min(r_ell) > 1.2:
                break

            # Create a subplot for the current frame
            ax = fig.add_subplot(projection='polar')

            # Set the title according to the time step
            ax.set_title(time_array[k].strftime('%Y-%m-%d %H:%M'))

            ax.plot(theta_ell, r_ell, color="tab:orange")

            # plot the position of the planets
            ax.scatter(mercury_lon, mercury_r, color = 'dimgrey', marker = 'o', label = 'Point')
            ax.scatter(venus_lon, venus_r, color = 'orange', marker = 'o', label = 'Point')
            ax.scatter(earth_lon, earth_r, color = 'mediumseagreen', marker = 'o', label = 'Point')
            ax.scatter(mars_lon, mars_r, color = 'orangered', marker = 'o', label = 'Point')

            # plot the position of the s/c
            if sta_available:
                ax.scatter(sta_lon, sta_r, color = 'red', marker = 's', label = 'Point')
            if stb_available:
                ax.scatter(stb_lon, stb_r, color = 'blue', marker = 's', label = 'Point')
            if mes_available:
                ax.scatter(mes_lon, mes_r, color = 'dimgrey', marker = 's', label = 'Point')
            if vex_available:
                ax.scatter(vex_lon, vex_r, color = 'orange', marker = 's', label = 'Point')
            if psp_available:
                ax.scatter(psp_lon, psp_r, color = 'black', marker = 's', label = 'Point')
            if solo_available:
                ax.scatter(solo_lon, solo_r, color = 'coral', marker = 's', label = 'Point')
            if bepi_available:
                ax.scatter(bepi_lon, bepi_r, color = 'blue', marker = 's', label = 'Point')

            if outer_system == 0:
                ax.set_rgrids((0.2, 0.4, 0.6, 0.8, 1.0), ('0.2', '0.4', '0.6', '0.8', '1 AU'), angle = 125, fontsize = 12,
                            alpha = 0.5, color = backcolor)
                ax.set_ylim(0, 1.2) 
            else:
                ax.scatter(jupiter_lon, jupiter_r, color = 'darkgoldenrod', marker = 'o', label = 'Point')
                ax.scatter(saturn_lon, saturn_r, color = 'palegreen', marker = 'o', label = 'Point')
                ax.scatter(uranus_lon, uranus_r, color = 'cyan', marker = 'o', label = 'Point')
                ax.scatter(neptune_lon, neptune_r, color = 'cornflowerblue', marker = 'o', label = 'Point')

                plt.rgrids((5, 10, 15, 20, 25, 30), ('5', '10', '15', '20', '25', '30 AU'), angle = 125, fontsize = 12,
                            alpha = 0.5, color = backcolor)
                ax.set_ylim(0, 32)
                
            #######################################################
            # shade HI field of view
            # HI1 field of view:
            hi1_fov_start = np.deg2rad(4.)
            hi1_fov_end = np.deg2rad(24.)

            #---- inner HI1 fov ------
            if HIobs == 'A':
                end_hi1_innerradius = np.sqrt(fov_length**2 + sta_r**2 - 2. * fov_length * sta_r * np.cos(hi1_fov_start))
                
                if sta_lon < 0:
                    end_hi1_innerangle = abs(np.arcsin((fov_length * np.sin(hi1_fov_start)) / end_hi1_innerradius) - np.pi) - abs(start_angle)                
                else:
                    end_hi1_innerangle = (abs(np.arcsin((fov_length * np.sin(hi1_fov_start)) / end_hi1_innerradius) - np.pi) - abs(start_angle)) * (-1)        
                
                line_x_hi1_inner = np.array([start_angle, end_hi1_innerangle])
                line_y_hi1_inner = np.array([start_radius, end_hi1_innerradius])
                #---- outer HI1 fov ------
                end_hi1_outerradius = np.sqrt(fov_length**2 + sta_r**2 - 2. * fov_length * sta_r * np.cos(hi1_fov_end))
                
                if sta_lon < 0:
                    end_hi1_outerangle = abs(np.arcsin((fov_length * np.sin(hi1_fov_end)) / end_hi1_outerradius)) - abs(start_angle)                
                else:
                    end_hi1_outerangle = (abs(np.arcsin((fov_length * np.sin(hi1_fov_end)) / end_hi1_outerradius)) - abs(start_angle)) * (-1)         
                    
                line_x_hi1_outer = np.array([start_angle, end_hi1_outerangle])
                line_y_hi1_outer = np.array([start_radius, end_hi1_outerradius])

                # HI2 field of view:
                hi2_fov_start = np.deg2rad(18.)
                hi2_fov_end = np.deg2rad(88.)

                #---- inner HI2 fov ------
                end_hi2_innerradius = np.sqrt(fov_length**2 + sta_r**2 - 2. * fov_length * sta_r * np.cos(hi2_fov_start))
                
                if sta_lon < 0:
                    end_hi2_innerangle = abs(np.arcsin((fov_length * np.sin(hi2_fov_start)) / end_hi2_innerradius) - np.pi) - abs(start_angle)                
                else:
                    end_hi2_innerangle = (abs(np.arcsin((fov_length * np.sin(hi2_fov_start)) / end_hi2_innerradius) - np.pi) - abs(start_angle)) * (-1)               
                
                line_x_hi2_inner = np.array([start_angle, end_hi2_innerangle])
                line_y_hi2_inner = np.array([start_radius, end_hi2_innerradius])
                
                #---- outer HI2 fov ------
                end_hi2_outerradius = np.sqrt(fov_length**2 + sta_r**2 - 2. * fov_length * sta_r * np.cos(hi2_fov_end))
            
                if sta_lon < 0:
                    end_hi2_outerangle = abs(np.arcsin((fov_length * np.sin(hi2_fov_end)) / end_hi2_outerradius)) - abs(start_angle)                
                else:
                    end_hi2_outerangle = (abs(np.arcsin((fov_length * np.sin(hi2_fov_end)) / end_hi2_outerradius)) - abs(start_angle)) * (-1)

            if HIobs == 'B':
                end_hi1_innerradius = np.sqrt(fov_length**2 + stb_r**2 - 2. * fov_length * stb_r * np.cos(hi1_fov_start))
                
                if stb_lon < 0:
                    end_hi1_innerangle = abs(np.arcsin((fov_length * np.sin(hi1_fov_start)) / end_hi1_innerradius) - np.pi) - abs(start_angle)                
                else:
                    end_hi1_innerangle = (abs(np.arcsin((fov_length * np.sin(hi1_fov_start)) / end_hi1_innerradius) - np.pi) - abs(start_angle)) * (-1)        
                
                line_x_hi1_inner = np.array([start_angle, end_hi1_innerangle])
                line_y_hi1_inner = np.array([start_radius, end_hi1_innerradius])
                #---- outer HI1 fov ------
                end_hi1_outerradius = np.sqrt(fov_length**2 + stb_r**2 - 2. * fov_length * stb_r * np.cos(hi1_fov_end))
                
                if stb_lon < 0:
                    end_hi1_outerangle = abs(np.arcsin((fov_length * np.sin(hi1_fov_end)) / end_hi1_outerradius)) - abs(start_angle)                
                else:
                    end_hi1_outerangle = (abs(np.arcsin((fov_length * np.sin(hi1_fov_end)) / end_hi1_outerradius)) - abs(start_angle)) * (-1)         
                    
                line_x_hi1_outer = np.array([start_angle, end_hi1_outerangle])
                line_y_hi1_outer = np.array([start_radius, end_hi1_outerradius])

                # HI2 field of view:
                hi2_fov_start = np.deg2rad(18.)
                hi2_fov_end = np.deg2rad(88.)

                #---- inner HI2 fov ------
                end_hi2_innerradius = np.sqrt(fov_length**2 + stb_r**2 - 2. * fov_length * stb_r * np.cos(hi2_fov_start))
                
                if stb_lon < 0:
                    end_hi2_innerangle = abs(np.arcsin((fov_length * np.sin(hi2_fov_start)) / end_hi2_innerradius) - np.pi) - abs(start_angle)                
                else:
                    end_hi2_innerangle = (abs(np.arcsin((fov_length * np.sin(hi2_fov_start)) / end_hi2_innerradius) - np.pi) - abs(start_angle)) * (-1)               
                
                line_x_hi2_inner = np.array([start_angle, end_hi2_innerangle])
                line_y_hi2_inner = np.array([start_radius, end_hi2_innerradius])
                
                #---- outer HI2 fov ------
                end_hi2_outerradius = np.sqrt(fov_length**2 + stb_r**2 - 2. * fov_length * stb_r * np.cos(hi2_fov_end))
            
                if stb_lon < 0:
                    end_hi2_outerangle = abs(np.arcsin((fov_length * np.sin(hi2_fov_end)) / end_hi2_outerradius)) - abs(start_angle)                
                else:
                    end_hi2_outerangle = (abs(np.arcsin((fov_length * np.sin(hi2_fov_end)) / end_hi2_outerradius)) - abs(start_angle)) * (-1)
                            
            line_x_hi2_outer = np.array([start_angle, end_hi2_outerangle])
            line_y_hi2_outer = np.array([start_radius, end_hi2_outerradius])

            # Shade the region between the two lines
            ax.plot(line_x_hi1_inner, line_y_hi1_inner, color='gray', linestyle='-', linewidth=2, alpha=0.1)
            ax.plot(line_x_hi1_outer, line_y_hi1_outer, color='gray', linestyle='-', linewidth=2, alpha=0.1)

            ax.plot(line_x_hi2_inner, line_y_hi2_inner, color='gray', linestyle='-', linewidth=2, alpha=0.1)
            ax.plot(line_x_hi2_outer, line_y_hi2_outer, color='gray', linestyle='-', linewidth=2, alpha=0.1)

            if k < len(elon_rad)-1:
                #print('k: ', k)
                ######
                # Calculate the ending point of the tangent
                if HIobs == 'A':

                    end_radius = np.sqrt(tangent_length**2 + sta_r**2 - 2. * tangent_length * sta_r * np.cos(elon_rad[k]))

                    # angle of end point of tangent
                    if np.cos(elon_rad[k]) > (sta_r/tangent_length): 
                        #print('version 1')
                        beta = np.arcsin((tangent_length * np.sin(elon_rad[k])) / end_radius) - np.pi
                    else:
                        #print('version 2')
                        beta = np.arcsin((tangent_length * np.sin(elon_rad[k])) / end_radius)

                    if sta_lon < 0:
                        end_angle = abs(beta) - abs(start_angle)
                    else:
                        end_angle = (abs(beta) - abs(start_angle)) * (-1)
                        
                if HIobs == 'B':

                    end_radius = np.sqrt(tangent_length**2 + stb_r**2 - 2. * tangent_length * stb_r * np.cos(elon_rad[k]))

                    # angle of end point of tangent
                    if np.cos(elon_rad[k]) > (stb_r/tangent_length): 
                        #print('version 1')
                        beta = np.arcsin((tangent_length * np.sin(elon_rad[k])) / end_radius) - np.pi
                    else:
                        #print('version 2')
                        beta = np.arcsin((tangent_length * np.sin(elon_rad[k])) / end_radius)

                    if stb_lon < 0:
                        end_angle = abs(beta) - abs(start_angle)
                    else:
                        end_angle = (abs(beta) - abs(start_angle)) * (-1)

                # Calculate the coordinates of the line
                line_x = np.array([start_angle, end_angle])
                line_y = np.array([start_radius, end_radius])
                # Plot the HI tangent
                ax.plot(line_x, line_y, color='red', linestyle='-', linewidth=2)
            
            if not movie:
                if not det_plot:
                    break
                
            if det_plot and k==200:
            #save single figure
                filename = prediction_path + 'ELEvoHI_HEE_m.jpg'
                plt.savefig(filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
                if not movie:
                    fig.clf()
                    plt.close(fig)
                    break
                
            if movie:                
            #save frames
                framestr = '%05i' % (k)
                filename = prediction_path + '/frames/frame_' + framestr + '.jpg' 
                plt.savefig(filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
                fig.clf()
                plt.close(fig)

        if movie:    
            os.system('ffmpeg -r 60 -i ' + prediction_path + '/frames/frame_%05d.jpg -b:v 5000k -r 60 ' + prediction_path + '/movie.mp4 -y')
    
    return prediction

def assess_prediction(prediction, target, is_time, is_speed):
# difference of prediction and in situ arrival
# Find the row where "target" is "L1" and access the "arrival time" 
    if isinstance(is_time, datetime):
        arrtime__target = prediction.loc[prediction['target'] == target, 'arrival time [UT]'].values[0]
        # Convert the "arrival time" to a datetime object
        arrtime_target = pd.to_datetime(arrtime__target)
        arrival_dt = (arrtime_target - is_time).total_seconds()/3600.
        
        prediction.loc[prediction['target'] == target, 'dt [h]'] = round(arrival_dt, 2)
        
        formatted_hours = "{:.2f}".format(arrival_dt)
        
        print('======================================================')
        print(f'ELEvoHI accuracy for {target}')

        print('------------------------------------------------------')

        if arrival_dt < 0:
            print('Predicted arrival is earlier than observed.')
            print('Difference: ', formatted_hours, 'hours')
        else:   
            print(' Predicted arrival is later than observed.')
            print('Difference: ', formatted_hours, 'hours')

        if isinstance(is_speed, int):
            arrspeed_target = prediction.loc[prediction['target'] == target, 'arrival speed [km/s]'].values[0]        
            arrival_dv = arrspeed_target - is_speed
            
            prediction.loc[prediction['target'] == target, 'dv [km/s]'] = int(round(arrival_dv))
            
            if arrival_dv < 0:
                print('Predicted arrival speed is lower than observed.')
                print('Difference: ', round(arrival_dv), ' km/s')
            else:
                print('Predicted arrival speed is higher than observed.')
                print('Difference: ', round(arrival_dv), ' km/s')

            print('======================================================')
        else:
            arrival_dv = np.nan
            print('======================================================')
    else:
            arrival_dt = np.nan
            arrival_dv = np.nan
        
    #pdb.set_trace()
            
    return arrival_dt, arrival_dv, prediction

def assess_ensemble(ensemble, det_results, det_run_no):
    
    ensemble_results = pd.DataFrame()
    
    target_names = ensemble['target'].unique()   
    number_of_runs = ensemble['run no.'].max() 
    
    j = 0
    
    for i in range(0, len(target_names)):
        if target_names[i] == 'No hit!':
            continue
        
        j = j + 1
        
        print('target_names: ', target_names[i])
        
        ensemble['arrival time [UT]'] = pd.to_datetime(ensemble['arrival time [UT]'])
        
        mean_arrival_time = ensemble[ensemble['target'] == target_names[i]]['arrival time [UT]'].mean()
        mean_arrival_time = mean_arrival_time.strftime("%Y-%m-%d %H:%M")
        median_arrival_time = ensemble[ensemble['target'] == target_names[i]]['arrival time [UT]'].median()
        median_arrival_time = median_arrival_time.strftime("%Y-%m-%d %H:%M")
        
        arrival_times_hours = ensemble[ensemble['target'] == target_names[i]]['arrival time [UT]'].dt.hour
        std_dev_hours = round(np.std(arrival_times_hours), 2)
        
        mean_arrival_speed = ensemble[ensemble['target'] == target_names[i]]['arrival speed [km/s]'].mean()
        median_arrival_speed = ensemble[ensemble['target'] == target_names[i]]['arrival speed [km/s]'].median()
        std_dev_speed = round(np.std(ensemble[ensemble['target'] == target_names[i]]['arrival speed [km/s]']))
        
        #Likelihood of hit at a certain target
        count = (ensemble['target'] == target_names[i]).sum()
        
        tmp_ensemble_results = pd.DataFrame(columns=['target'])
        tmp_ensemble_results.loc[j, 'target'] = target_names[i]
        tmp_ensemble_results['likelihood [%]'] = round(count/(number_of_runs * 0.01))
        
        if (det_results['target'] == target_names[i]).any():
            tmp_ensemble_results['arrival time (det) [UT]'] = det_results['arrival time [UT]'][0]
            tmp_ensemble_results['arrival speed (det) [km/s]'] = det_results['arrival speed [km/s]'][0]
            #pdb.set_trace()
        else:
            tmp_ensemble_results['arrival time (det) [UT]'] = 'No hit!'
            tmp_ensemble_results['arrival speed (det) [km/s]'] = np.nan
                        
        tmp_ensemble_results['arrival time (mean) [UT]'] = mean_arrival_time
        tmp_ensemble_results['arrival time (median) [UT]'] = median_arrival_time
        tmp_ensemble_results['arrival time (std dev) [h]'] = std_dev_hours
        tmp_ensemble_results['arrival speed (mean) [km/s]'] = round(mean_arrival_speed)
        tmp_ensemble_results['arrival speed (median) [km/s]'] = round(median_arrival_speed)
        tmp_ensemble_results['arrival speed (std dev) [km/s]'] = std_dev_speed
        
        tmp_ensemble_results['ME (t)'] = round(ensemble[ensemble['target'] == target_names[i]]['dt [h]'].mean(), 2)       
        tmp_ensemble_results['MAE (t)'] = round(ensemble[ensemble['target'] == target_names[i]]['dt [h]'].abs().mean(), 2)   
        tmp_ensemble_results['RMSE (t)'] = round(np.sqrt(((ensemble[ensemble['target'] == target_names[i]]['dt [h]'])**2).mean()), 2)
        
        tmp_ensemble_results['ME (v)'] = round(ensemble[ensemble['target'] == target_names[i]]['dv [km/s]'].mean(), 2)       
        tmp_ensemble_results['MAE (v)'] = round(ensemble[ensemble['target'] == target_names[i]]['dv [km/s]'].abs().mean(), 2)   
        tmp_ensemble_results['RMSE (v)'] = round(np.sqrt(((ensemble[ensemble['target'] == target_names[i]]['dv [km/s]'])**2).mean()), 2)
                       
        ensemble_results = pd.concat([ensemble_results, tmp_ensemble_results])
        
    
    #pdb.set_trace()
    
    return ensemble_results
