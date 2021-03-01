#This file finds the conjunctions between SSUSI and SSJ observations across the time frame studied.

import datetime
from geospacepy import special_datetime, satplottools
from geospacepy.special_datetime import (datetimearr2jd,
                                        datetime2jd,
                                        jd2datetime)
import os, glob, string, numpy as np
import pdb
import scipy
import matplotlib as mpl
import matplotlib.pyplot as pp
from matplotlib.gridspec import GridSpec
import sys
import time
import h5py
from numpy import (sin,cos,tan,arcsin,arccos,arctan2)
from copy import copy, deepcopy
from LBH_to_eflux.observations.ssj import SSJDay
from LBH_to_eflux.observations.ssusi import SDRPass
from sklearn.neighbors import NearestNeighbors


class SSUSIandSSJConjunctions(object):

    @staticmethod
    def get_conjunction_data(ssusi_pass_obs,ssj_pass_obs, obs_to_interpolate = ['LBHL', 'LBHS'], k = 10, tol = 1.5):
        """
        Performs K nearest neighbors distance weighted average and returns dictionary of SSUSI-SSJ conjunction data for a given SSUSI and SSJ data


        Parameters 
        ----------
        ssusi_pass_obs : dict object with elements
            ['jds'] : np.ndarray (n_obs x 1)
                Array of observation times (in Julian Date)
            ['lats'] : np.ndarray (n_obs x 1)
                Magnetic latitude (degrees) of observations in Apex coordinates of reference height 110 km 
            ['lons'] : np.ndarray (n_obs x 1)
                Magnetic local time of observations (expressed in degrees ) of observations in Apex coordinates 

            Observations of the form 
            ['LBHL']  : np.ndarray (n_obs x 1)
            ['LBHS']  : np.ndarray (n_obs x 1)
            ['Lyman Alpha']  : np.ndarray (n_obs x 1)
            ['OI 130.4']  : np.ndarray (n_obs x 1)
            ['OI 135.6']  : np.ndarray (n_obs x 1)

        ssj_pass_obs : dict object 
            Same dict as output from SSJDay get_ingest_data() 
        obs_to_interpolate : list of str 
            Needs to be subset of ['Lyman Alpha','OI 130.4', 'OI 135.6', 'LBHS', 'LBHL']
            Defaults to ['LBHL', 'LBHS']
        k : int 
            Neighbor of SSUSI pixels to include in distance weighted averaging
        tol : float 
            Tolerated distance (in degrees) from SSUSI obs to SSJ obs to be considered in a given conjunction statistic

        Returns
        -------
        conjunction_dict : dict object 
            Same elements as ssj_pass_obs with interpolated SSUSI obs values on the SSJ track
            eg. ['LBHL_interped']
        """
        conjunction_dict = deepcopy(ssj_pass_obs)

        #iterate across observations to interp and put them in a dictionary 
        for obs_name in obs_to_interpolate:
            conjunction_dict[obs_name + '_interped'] = SSUSIandSSJConjunctions.dmsp_map_interpolate_NN_smooth_great_circle(ssj_pass_obs['lats'], ssj_pass_obs['lons'], \
                                                                            ssusi_pass_obs['lats'], ssusi_pass_obs['lons'], ssusi_pass_obs[obs_name], \
                                                                            k = k, tol = tol)

        return conjunction_dict

    @staticmethod
    def write_to_h5(hd_dir, datadict, hemi, dmsp, dt):
        """
        Saves the conjucntion dictionary as a h5 file 
        """
        file_dir = os.path.join(hd_dir,'SSUSI_SSJ_conjunctions_{}{}{}.hdf5'.format(hemi,dmsp,dt.strftime('%Y%m%d_%H:%M')))

        #write data dictionary to h5 file 
        h5f = h5py.File(file_dir,'w')
        for key in datadict:
            h5f.create_dataset(key,data = datadict[key])
        h5f.close()
        return
    
    @staticmethod
    def dmsp_map_interpolate_NN_smooth_great_circle(lat_dmsp, lon_dmsp, lat_map, lon_map, Obs_map, k = 5, tol = 1.5):
        """
        Wrapper for Sklearn's nearestneighbor with haversine distance 
        """
        tol = np.deg2rad(tol)
        #reshape to N by 2 array where each row is (lat, lon)
        dmsp_points = np.deg2rad(np.hstack((lat_dmsp.flatten().reshape(-1,1),lon_dmsp.flatten().reshape(-1,1))))
        map_points = np.deg2rad(np.hstack((lat_map.flatten().reshape(-1,1), lon_map.flatten().reshape(-1,1))))
        N_points = dmsp_points.shape[0]
        obs_val = Obs_map.flatten()

        model = NearestNeighbors(n_neighbors = k, radius = tol, metric = 'haversine')

        model.fit(map_points)
        neighbors = model.kneighbors(dmsp_points, return_distance = True)
        
        #indices
        obs_interp = np.empty(N_points)
        for i in range(N_points):
            distances = neighbors[0][i]
            inds = neighbors[1][i]
            
            weights = distances/np.nansum(distances)
            obs_interp[i] = np.nansum( obs_val[inds] * weights)
            
            
        return obs_interp


#Feel free to change this to whatever timeframe you need 
dmsp_arr = [16,17,18] #satellites to run for 
year = 2014
day_arr = [17,18,19,20,21,22,23] #days of month to run for  
doy_arr = [48,49,50,51,52,53,54]
# doy_arr = [54]

hemis = ['N','S']
radiances_to_use = ['LBHL','LBHS','Lyman Alpha']
#make sure to change this 
ssusi_dir = os.path.join('/home/matsuo/amgeo_dev/LBH_to_eflux/LBH_to_eflux/','SSUSI_SDR_Observations')
ssj_dir = os.path.join('/home/matsuo/amgeo_dev/LBH_to_eflux/LBH_to_eflux/','SSJ_Observations')

#place you would like to save the conjunction files 
conjunction_dir = '/home/matsuo/amgeo_dev/LBH_to_eflux/LBH_to_eflux/conjunction_data'
if not os.path.exists(conjunction_dir):
    os.mkdir(conjunction_dir)

passnumber = 0
for doy in doy_arr:
    ssj_day_dir = ssj_dir + '/{}{}'.format(year,doy)
    ssusi_day_dir = ssusi_dir +  '/{}{}'.format(year,doy)

    for dmsp in dmsp_arr:

        #read the SSJ obs file
        day_ssj_file =  glob.glob(os.path.join(ssj_day_dir,'dmsp-f%d_ssj_*.cdf' % (dmsp)))[0]
        ssj_obs = SSJDay(dmsp,'N',day_ssj_file, read_spec = True)

        #get all the ssusi files
        ssusi_files = glob.glob(os.path.join(ssusi_day_dir, 'dmspf%d_ssusi_*' % (dmsp)))
        ssusi_files = np.sort(ssusi_files) 

        #iterate across the ssusi files
        for ssusi_file in ssusi_files:
            #convert file to byte literal
            ssusi_file = bytes(ssusi_file.encode())

            #get the observations for each file
            ssusi_obs_LBHL = SDRPass(ssusi_file, dmsp, 'N','LBHL')
            ssusi_obs_LBHS = SDRPass(ssusi_file, dmsp, 'N','LBHS')
            ssusi_obs_LYMAN = SDRPass(ssusi_file,dmsp, 'N','Lyman Alpha')

            #start and end times of this pass
            startdt = jd2datetime(np.nanmin(ssusi_obs_LBHL['jds']))
            enddt = jd2datetime(np.nanmax(ssusi_obs_LBHL['jds']))

            for hemi in hemis:
                try:
                    ssusi_lats,ssusi_lons,LBHL,LBHL_var,jds = ssusi_obs_LBHL.get_ingest_data(hemisphere = hemi)
                    ssusi_lats,ssusi_lons,LBHS,LBHS_var,jds = ssusi_obs_LBHS.get_ingest_data(hemisphere = hemi)
                    ssusi_lats,ssusi_lons,LYMAN,LYMAN_var,jds = ssusi_obs_LBHS.get_ingest_data(hemisphere = hemi)

                    ssusi_pass_obs = {}
                    ssusi_pass_obs['lats'], ssusi_pass_obs['lons'], ssusi_pass_obs['jds']= ssusi_lats, ssusi_lons, jds
                    ssusi_pass_obs['LBHL'], ssusi_pass_obs['LBHL_var'] = LBHL, LBHL_var
                    ssusi_pass_obs['LBHS'], ssusi_pass_obs['LBHS_var'] = LBHS, LBHS_var
                    ssusi_pass_obs['LYMAN'], ssusi_pass_obs['LYMAN_var'] = LYMAN, LYMAN_var

                    #get the relevant SSJ data for this pass 
                    ssj_pass_obs = ssj_obs.get_ingest_data(startdt = startdt, enddt = enddt, hemisphere = hemi)

                    # if ssusi obs overlap to next day, read next day ssj data and append to ssj_pass_obs dict
                    if enddt > np.nanmax(ssj_obs['epoch']):

                        ssj_next_day_dir = ssj_dir + '/{}{}'.format(year,doy+1)
                        next_day_ssj_file =  glob.glob(os.path.join(ssj_next_day_dir,'dmsp-f%d_ssj_*.cdf' % (dmsp)))[0]
                        ssj_obs_next_day = SSJDay(dmsp,'N',next_day_ssj_file, read_spec = True)
                        ssj_pass_obs_next_day = ssj_obs_next_day.get_ingest_data(startdt = startdt, enddt = enddt, hemisphere = hemi)

                        #append to ssj_pass _obs
                        for key in ssj_pass_obs:
                            ssj_pass_obs[key] =  np.append(ssj_pass_obs[key], ssj_pass_obs_next_day[key])                
                    
                    #get times for the ssusi pass
                    pass_startdt = jd2datetime(np.nanmin(ssj_pass_obs['jds']))
                    pass_enddt = jd2datetime(np.nanmax(ssj_pass_obs['jds']))
                    pass_center_dt = pass_startdt + (pass_enddt - pass_startdt)/2
                    print(pass_center_dt)

                    del ssj_pass_obs['epoch']

                    #get the conjunctions between SSUSI and SSJ 
                    conjunctions = SSUSIandSSJConjunctions.get_conjunction_data(ssusi_pass_obs, ssj_pass_obs, obs_to_interpolate = ['LBHL','LBHS','LYMAN'], k = 10, tol = 1)
                    conjunctions['pass_num'] = np.ones_like(conjunctions['jds']) * passnumber
                    conjunctions['sat_no'] = np.ones_like(conjunctions['jds']) * dmsp
                    if hemi == 'N':
                        conjunctions['hemi'] = np.ones_like(conjunctions['jds'])
                    else:
                        conjunctions['hemi'] = np.ones_like(conjunctions['jds']) * -1
 
                    #save to h5 file
                    SSUSIandSSJConjunctions.write_to_h5(conjunction_dir, conjunctions, hemi, dmsp, pass_center_dt)

                    passnumber+=1
                except:
                    print('Invalid conjunctions')



