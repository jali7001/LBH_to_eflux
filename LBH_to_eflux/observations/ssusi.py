from collections import OrderedDict
import numpy as np
from geospacepy.special_datetime import (doyarr2datetime,
                                        datetimearr2jd,
                                        datetime2jd,
                                        jd2datetime,
                                        datetime2doy,
                                        jdarr2datetime)
from geospacepy import special_datetime
from geospacepy.terrestrial_spherical import (eci2ecef,
                                            ecef_cart2spherical,
                                            ecef_spherical2cart,
                                            ecef2enu)
from geospacepy.terrestrial_ellipsoidal import ecef_cart2geodetic
from geospacepy.satplottools import latlon2cart
import esabin
import datetime, os, timeit, random
from netCDF4 import Dataset
from collections import OrderedDict
import h5py
import glob
from sklearn import linear_model

from ..helper_funcs import latlt2polar,polar2dial, module_Apex, update_apex_epoch


class ssusiSDRpass(object):
    """
    
    Attributes
    ----------
    noise_removal - 
    spatial_bin
    minlat - 
    ssusi_colors 
    arg_radiance 
    year
    month
    day
    spatial_bin
    noise_removal


    """
    def __init__(self, ssusi_file, dmsp, hemisphere, radiance_type = 'LBHL', noise_removal = True, spatial_bin = True, ):
        """
        
        Inputs
        ------
        Hemisphere

        """
        self.noise_removal = noise_removal
        self.spatial_bin = spatial_bin 
        self.minlat = 50
        self.ssusi_colors = {
            'Lyman Alpha':0,
            'OI  130.4':1,
            'OI 135.6':2,
            'LBHS':3,
            'LBHL':4
        }
        self.arg_radiance = self.ssusi_colors[radiance_type]

        self.name = 'SSUSI ' + radiance_type
        self.hemisphere = hemisphere

        self.radiance_type = radiance_type
        self.spatial_bin = spatial_bin

        # self.year,self.month,self.day = dt.year,dt.month,dt.day

        #prepare grid if spatially binning
        if spatial_bin:
            self.grid = esabin.esagrid.Esagrid(2, azi_coord = 'lt')
        
        pass_data = self.get_ssusi_pass(ssusi_file, dmsp)

        self['jds'] = pass_data['epochjd_match']
        self['observer_ids'] = pass_data['observer']

        self['Y'] = pass_data['Y']
        self['Y_var'] = pass_data['Y_var']
        self['lats'] = pass_data['mlat']
        self['lons'] = pass_data['mlon']

    def get_data_window(self, startdt, enddt, hemisphere, allowed_observers):
        """
        
        """
        mask = self.get_data_window_mask(startdt, enddt, hemisphere, allowed_observers)

        data_window = OrderedDict()
        for datavarname,datavararr in self.items():
            data_window[datavarname] = datavararr[mask]

        return data_window


    def _read_SDR_file(self, file_name):
        """
        Reads in the Disk Radiances and their piercepoint day observation location.
        Returns Disk dictionary object
        
        """
        with h5py.File(file_name,'r') as h5f:

            disk = {} #initialize disk measurements
            disk['glat'] = h5f['PIERCEPOINT_DAY_LATITUDE_AURORAL'][:]
            disk['glon'] = h5f['PIERCEPOINT_DAY_LONGITUDE_AURORAL'][:]
            disk['alt'] = h5f['PIERCEPOINT_DAY_ALTITUDE_AURORAL'][:]
            disk['time'] = h5f['TIME_DAY_AURORAL'][:] #seconds since start of day
            disk['year'] = h5f['YEAR_DAY_AURORAL'][:] 
            disk['doy'] = h5f['DOY_DAY_AURORAL'][:]
            disk['SZA'] = h5f['PIERCEPOINT_DAY_SZA_AURORAL'][:]


            #read radiances as kR
            disk['radiance_all_colors_uncertainty'] = h5f['DISK_RADIANCE_UNCERTAINTY_DAY_AURORAL'][:] /1000.
            disk['radiance_all_colors'] = h5f['DISK_RECTIFIED_INTENSITY_DAY_AURORAL'][:] / 1000.


        #get epoch from seconds of day, day of year, and year in terms of datetimes 
        dt = np.empty((len(disk['doy']),1),dtype='object')
        for k in range(len(dt)):
            dt[k,0] = datetime.datetime(disk['year'][k],1,1,0,0,0)+datetime.timedelta(days=disk['doy'][k]-1.) + datetime.timedelta(seconds= disk['time'][k])
        disk['epoch'] = dt.flatten()        

        return disk

    def get_ssusi_pass(self, ssusi_file, dmsp):
        """
        Main function to read in and preprocess SDR file. 
        1. Read SDR ncdf file
        2. Solar influence removal
        3. Spatial binning if desired
        """

        #read in each file 
        disk_data = self._read_SDR_file(ssusi_file)
        
        #integrate disk data into usable magnetic coordinates
        disk_int = self._ssusi_integrate(disk_data)

        #report mlon is magnetic local time in degrees 
        disk_int['mlon'] = disk_int['mlt'] * 15

        #get observation times
        shape = np.shape(disk_int['mlat'])
        disk_int['epoch_match'] = np.tile(disk_int['epoch'].flatten(), (shape[0],1))

        #get the relevant observations
        disk_int['Y'] = disk_int['radiance_all_colors'][:,:,self.arg_radiance]
        disk_int['Y_var'] = disk_int['radiance_all_colors_uncertainty'][:,:,self.arg_radiance]

        #get rid of negative values 
        disk_int['Y'][disk_int['Y']<0] = 0

        #if solar influence removal
        if self.noise_removal:
            radiance_fit = self._radiance_zenith_correction(disk_data['SZA'],disk_int['Y'])                    
            disk_int['Y'] = disk_int['Y'] - radiance_fit

        #flatten data
        for item in disk_int:
            disk_int[item] = disk_int[item].flatten()

        #get times in terms of jds
        disk_int['epochjd_match'] = datetimearr2jd(disk_int['epoch_match']).flatten()

        #Spatially bin observations if desired
        if self.spatial_bin:
            disk_int = self.ssusi_spatial_binning(disk_int)

        disk_int['observer'] = np.ones_like(disk_int['epochjd_match']) * dmsp

        return disk_int 

    def ssusi_spatial_binning(self,disk_int):
        """
        This function spatially bins the 
        """

        disk_binned = {}
        #spatial bin 
        lats_in_pass, lts_in_pass, Y_in_pass, Y_var_in_pass = disk_int['mlat'], disk_int['mlt'], disk_int['Y'], disk_int['Y_var']
        epochjds_in_pass = disk_int['epochjd_match']
        
        #convert from mlt [0, 24] to [-12, 12]
        lts_mask = lts_in_pass >= 12
        lts_in_pass[lts_mask] -= 24 

        binlats, binlons, binstats = self.grid.bin_stats(lats_in_pass.flatten(), lts_in_pass.flatten(), Y_in_pass.flatten(), \
                                            statfun = np.nanmean, center_or_edges = 'center')

        binlats, binlons, binstats_var = self.grid.bin_stats(lats_in_pass.flatten(), lts_in_pass.flatten(), Y_var_in_pass.flatten(), \
                                    statfun = np.nanvar, center_or_edges = 'center')

        binlats, binlons, binstats_time = self.grid.bin_stats(lats_in_pass.flatten(), lts_in_pass.flatten(), epochjds_in_pass.flatten(), \
                                            statfun = np.nanmedian, center_or_edges = 'center')

        #convert from mlt -12 to 12 to degrees 0 to 360
        binlons[binlons>=0] = 15*binlons[binlons>=0]
        binlons[binlons<0] = (binlons[binlons<0]+24)*15


        disk_binned['mlat'], disk_binned['mlon'], disk_binned['Y'], disk_binned['Y_var'] = binlats, binlons, binstats, binstats_var
        disk_binned['epochjd_match'] = binstats_time
        return disk_binned

    def geo2apex(self,datadict):
        """
        Perform coordinate transform for disk measurements from geographic to apex magnetic coordinates
        """
        #take coordinates from data dictionary
        glat,glon,alt = datadict['glat'].flatten(),datadict['glon'].flatten(),datadict['alt'][0]
        alt = 110
        dt_arrs = datadict['epoch']

        #convert to apex coordinates
        alat = np.full_like(glat,np.nan)
        alon = np.full_like(glat,np.nan)
        qdlat = np.full_like(glat,np.nan)
        
        update_apex_epoch(dt_arrs[0]) #This does not need to be precise, time-wise

        alatout,alonout = module_Apex.geo2apex(glat,glon,alt)
        alat,alon = alatout.flatten(),alonout.flatten()
        
        #calculate time for observations because it isn't available in SDR product
        utsec = datadict['time'].flatten()
        utsec = (np.tile(utsec, (42,1))).flatten()

        dt_arrs_tiled = (np.tile(dt_arrs, (42,1))).flatten()
        mlt = np.full_like(alon, np.nan)

        for i in range(np.size(mlt)):
            mlt[i] = module_Apex.mlon2mlt(alon[i], dt_arrs_tiled[i],alt)

        #reshape to original shapes
        alat = alat.reshape(datadict['glat'].shape)
        mlt = mlt.reshape(datadict['glon'].shape)

        return alat,mlt  

    def get_ingest_data(self,startdt, enddt, hemisphere, allowed_observers):
        """
        Returns
        -------
        lats : np.ndarray
            Absolute magnetic latitudes of observations
        lons : np.ndarray
            magnetic 'longitudes' (MLT in degrees) of observations
        y : np.ndarray
            1D array of kiloRayleighs
        y_var : np.ndarray
            1D array of uncertainies (variances) in magnetic perturbations
            (i.e. diagonal of observation error covariance matrix)
        """
        startdt = jd2datetime(np.nanmin(self['jds']))
        enddt = jd2datetime(np.nanmax(self['jds']))
        datadict = self.get_data_window(startdt,
                                        enddt,
                                        hemisphere,
                                        allowed_observers)
        y = datadict['Y'].reshape(-1,1)


        #Format the error/variance vector similarly,
        y_var = datadict['Y_var'].reshape(-1,1);

        #Locations for each vector component
        ylats = datadict['lats'].reshape(-1,1)
        ylons = datadict['lons'].reshape(-1,1)

        #jds 
        jds = datadict['jds'].reshape(-1,1)
        return np.abs(ylats),ylons,y,y_var,jds 

    def plot_obs(self, ax, startdt,enddt,hemisphere,allowed_observers, **kwargs):

        lats,lons,obs,y_var,jds = self.get_ingest_data(startdt,enddt,hemisphere,allowed_observers)
        r,theta = latlt2polar(lats.flatten(),lons.flatten()/180*12,'N')

        ax.scatter(theta,r,c = obs, **kwargs)

        polar2dial(ax)

    @staticmethod
    def _ssusi_integrate_position(position_data):
        return np.squeeze(position_data[:,:])

    @staticmethod
    def _ssusi_integrate_radiance(radiance_data):
        return np.squeeze(radiance_data[:,:])

    def _ssusi_integrate(self,datadict):
        """
        General wrapper for coordinate convserion
        datadict - dict
            use the output dictionary from the read in function readSSUSI1bdata()
        limb_or_disk - str
            must be 'limb' or 'disk' for the appropriate observations 
        """
        #Disk dimension N x 132 x 16 (N scans x M integration steps x P pixels)
        #Limb dimension N x 24 x 8 x 5 (N scans x M integration steps x P pixels)
        datadict_out = {}
        
        for varname in datadict:
            if 'radiance' in varname:
                datadict_out[varname] = self._ssusi_integrate_radiance(datadict[varname])
            elif varname in ['glat','glon','SZA']:
                datadict_out[varname] = self._ssusi_integrate_position(datadict[varname])
            else:
                datadict_out[varname] = datadict[varname]

        alat,mlt = self.geo2apex(datadict_out)

        datadict_out['mlat'] = alat
        datadict_out['mlt'] = mlt

        return datadict_out

    @staticmethod
    def _radiance_zenith_correction(sza,radiance):
        """
        A quick correction for the solar influence noise on the radiance data using a simple regression following the methods of 

        SZA - list, num_obsx1
            solar zenith angle of the DMSP spacecraft (degrees)
        radiance - list, num_obsx1
            radiance observations 
        """

        finite = np.logical_and(np.isfinite(sza.flatten()),
                                np.isfinite(radiance.flatten()))

        clf = linear_model.LinearRegression(fit_intercept=True)
        X = sza.reshape((-1,1))
        X = np.cos(np.deg2rad(sza).reshape((-1,1)))
        y = radiance.reshape((-1,1))
        clf.fit(X[finite],y[finite])

        return clf.predict(X).reshape(radiance.shape)


    def __str__(self):
        return '{} {}:\n hemisphere {},\n date {}-{}-{}'.format(self.name,
                                                        self.observation_type,
                                                        self.hemisphere,
                                                        self.year,
                                                        self.month,
                                                        self.day)

    def __setitem__(self,item,value):
        if not hasattr(self,'_observation_data'):
            self._observation_data = OrderedDict()
        self._observation_data[item]=value

    def __getitem__(self,item):
        return self._observation_data[item]

    def __contains__(self,item):
        return item in self._observation_data

    def __iter__(self):
        for item in self._observation_data:
            yield item

    def items(self):
        for key,value in self._observation_data.items():
            yield key,value

    def get_data_window_mask(self,startdt,enddt,hemisphere,allowed_observers):
        """Return a 1D mask into the data arrays for all points in the
        specified time range, hemisphere and with radar IDs (RIDs) in
        list of RIDs allowed_observers"""
        mask = np.ones(self['jds'].shape,dtype=bool)
        mask = np.logical_and(mask, self._get_finite())
        mask = np.logical_and(mask,self._get_time_mask(startdt,enddt))
        mask = np.logical_and(mask,self._get_hemisphere_mask(hemisphere))
        mask = np.logical_and(mask,self._get_observers_mask(allowed_observers))
        return mask

    def _get_finite(self):
        return np.isfinite(self['Y'])

    def _get_time_mask(self,startdt,enddt):
        """Return mask in data arrays for all
        data in interval [startdt,enddt)"""
        startjd = special_datetime.datetime2jd(startdt)
        endjd = special_datetime.datetime2jd(enddt)
        if endjd< np.nanmin(self['jds']) or startjd> np.nanmax(self['jds']):
            dts = (startdt,enddt)
            raise DataWindowBoundaryError(('Data required for: '
                                          +self._window_time_bounds_str(*dts)
                                          +'out of range for SuperDARN object: '
                                          +self._time_bounds_str()))
        inrange = np.logical_and(self['jds'].flatten()>=startjd,
                                    self['jds'].flatten()<endjd)
        return inrange

    def _get_hemisphere_mask(self,hemisphere):

        if hemisphere not in ['N','S']:
            return ValueError(('{}'.format(hemisphere)
                              +' is not a valid hemisphere (use N or S)'))
        if hemisphere != self.hemisphere:
            raise DataWindowBoundaryError('hemisphere {}'.format(hemisphere)
                                          +' was requested but Ampere'
                                          +' object only contains data for'
                                          +' {}'.format(self.hemisphere)
                                          +' hemisphere.')
        if hemisphere == 'N':
            inhemi = self['lats'] > self.minlat
        elif hemisphere == 'S':
            inhemi = self['lats'] < -self.minlat
        return inhemi
    def _check_allowed_observers(self,allowed_observers):
        if not isinstance(allowed_observers,list):
            raise RuntimeError('allowed_observers must be a list of '
                               +'ampere pseudo vehicle numbers')
        for observer_id in allowed_observers:
            if observer_id not in self.observers:
                raise ValueError('Pseudo vehicle number {} not'.format(observer_id)
                                 +'in data PVNs \n({})'.format(self.observers))

    def _get_observers_mask(self,allowed_observers):
        if allowed_observers != 'all':
            self._check_allowed_observers(allowed_observers)

            observers_mask = np.zeros(self['jds'].shape,dtype=bool)
            for observer_id in allowed_observers:
                observers_mask = np.logical_or(observers_mask,
                                                self['observer_ids']==observer_id)
        else:
            observers_mask = np.ones(self['jds'].shape,dtype=bool)
            
        return observers_mask
 