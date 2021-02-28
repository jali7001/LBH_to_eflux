from collections import OrderedDict
import numpy as np
from geospacepy.special_datetime import (datetimearr2jd,
                                        datetime2jd,
                                        jd2datetime)
import esabin
import datetime, os
import h5py
from sklearn import linear_model
from LBH_to_eflux.helper_funcs import latlt2polar, polar2dial, module_Apex, update_apex_epoch


class SDRPass(object):
    """
    Description
    -----------
    This class reads in a NASA CDAweb SSUSI SDR disk file corresponding to one spacecraft orbit and stores the pass data as class attributes.
    After instantiating this class, you can call get_ingest_data() to get observations for one polar pass,

    Attributes
    ----------
    name : str
        Name of observations
    ['jds'] : np.ndarray (n_obs x 1)
        Array of observation times (in Julian Data) 
    ['observer_ids'] : np.ndarray (n_obs x 1)
        Satellite number associated with observations
    ['Y'] : array_like (n_obs x 1)
        Observations of FUV radiance whose color is specified by radiance_type  (in kilo rayleighs)
    ['Y_var'] : np.ndarray (n_obs x 1)
        Observation error 
    ['lats'] : np.ndarray (n_obs x 1)
        Magnetic latitude (degrees) of observations in Apex coordinates of reference height 110 km 
    ['lons'] : np.ndarray (n_obs x 1)
        Magnetic local time of observations (expressed in degrees ) of observations in Apex coordinates 
    ['observer_ids'] : np.ndarrray (n_obs x 1)
        Satellite number associated with observations 
    """
    def __init__(self, ssusi_file, dmsp, hemisphere, radiance_type = 'LBHL', noise_removal = True, spatial_bin = False, minlat = 50):
        """
        Parameters
        ----------
        ssusi_file : str
            Location of SSUSI file to read in
        dmsp : int 
            DMSP spacecraft number which must take value in [16, 17, 18]
        hemisphere : str
            Hemisphere of observations (must be either 'N' or 'S')
        radiance_type : str
            "Color" of FUV radiance to read in. Must be one of ['Lyman Alpha','OI 130.4', 'OI 135.6', 'LBHS', 'LBHL']
        noise_removal : bool, optional
            If true, removes solar influence from FUV radiances 
            Its default value is True
        spatial_bin : bool, optional
            If true, spatially bins observations
            Its default value is False
        minlat : int, optional 
            Minimum latitude in magnetic degrees  a polar pass 
        """
        self.ssusi_file = ssusi_file
        self.dmsp = dmsp
        self.hemisphere = hemisphere
        self.radiance_type = radiance_type
        self.noise_removal = noise_removal
        self.spatial_bin = spatial_bin 
        self.minlat = minlat


        self.name = 'SSUSI ' + radiance_type
        self.ssusi_colors = {
            'Lyman Alpha':0,
            'OI  130.4':1,
            'OI 135.6':2,
            'LBHS':3,
            'LBHL':4
        }
        self.observers = [16,17,18]

        self.arg_radiance = self.ssusi_colors[radiance_type]

        #prepare grid if spatially binning
        if spatial_bin:
            self.grid = esabin.esagrid.Esagrid(2, azi_coord = 'lt')
        
        pass_data = self.get_ssusi_pass()

        self['jds'] = pass_data['epochjd_match']
        self['observer_ids'] = pass_data['observer']

        self['Y'] = pass_data['Y']
        self['Y_var'] = pass_data['Y_var']
        self['lats'] = pass_data['mlat']
        self['lons'] = pass_data['mlon']

    def get_data_window(self, startdt, enddt, hemisphere, allowed_observers):
        """
        Applies the hemisphere and datetime interval to the attributes of the class.
        
        Parameters
        ----------
        startdt : datetime object
            Defaults to first available observation time
        enddt : datetime object
            Defaults to last available observation time
        hemisphere : str
            Defaults to hemisphere specified in init 
    
        Returns
        -------
        data_window : OrderedDict
            Contains the following elements
            ['jds'] : np.ndarray (n_obs x 1)
                Array of observation times (in Julian Data) 
            ['observer_ids'] : np.ndarray (n_obs x 1)
                Satellite number associated with observations
            ['Y'] : array_like (n_obs x 1)
                Observations of FUV radiance whose color is specified by radiance_type  (in kilo rayleighs)
            ['Y_var'] : np.ndarray (n_obs x 1)
                Observation error 
            ['lats'] : np.ndarray (n_obs x 1)
                Magnetic latitude (degrees) of observations in Apex coordinates of reference height 110 km 
            ['lons'] : np.ndarray (n_obs x 1)
                Magnetic local time of observations (expressed in degrees ) of observations in Apex coordinates 
            ['observer_ids'] : np.ndarrray (n_obs x 1)
                Satellite number associated with observations 

        """
        mask = self.get_data_window_mask(startdt, enddt, hemisphere, allowed_observers)

        data_window = OrderedDict()
        for datavarname,datavararr in self.items():
            data_window[datavarname] = datavararr[mask]

        return data_window


    def _read_SDR_file(self, file_name):
        """
        Reads in the Disk Radiances and their piercepoint day observation location. 
        SSUSI data comes in sweeps of 42 cross track observations.
        Therefore the total number of observation is n_obs = 42 * n_sweeps

        Parameters
        ----------
        file_name : str
            location of SSUSI SDR file 
        Returns
        -------
        disk : dict
            Dictionary of relevant values from the SDR value with elements
            ['glat'] : np.ndarray (42 x n_sweeps)
                Geographic Latitude of observations
            ['glon'] : np.ndarray (42 x n_sweeps)
                Geographic longitude of observations
            ['alt'] : np.ndarray (1 x 1)
                Altitude of observations
            ['time'] : np.ndarray (n_sweeps)
                Seconds during day
            ['year'] : np.ndarray (n_sweeps)
                Year of obs
            ['radiance_all_colors']  : np.ndarray (42 x n_sweeps x 5)
                All 5 FUV "colors" (kRa)
            ['radiance_all_colors_uncertainty']  : np.ndarray (42 x n_obs x 5)
                Uncertainty of FUV radiances (kRa)
            ['SZA'] : np.ndarray  (42 x n_sweeps x 5)
                Solar Zenith angle of observations (deg)
            ['epooch'] : list (n_sweeps x 1)
                list of observation times in datetime objects

        """
        disk = {} #initialize disk measurements

        with h5py.File(file_name,'r') as h5f:
            #location of obs in geographic coordinates
            disk['glat'] = h5f['PIERCEPOINT_DAY_LATITUDE_AURORAL'][:] 
            disk['glon'] = h5f['PIERCEPOINT_DAY_LONGITUDE_AURORAL'][:]
            disk['alt'] = h5f['PIERCEPOINT_DAY_ALTITUDE_AURORAL'][:]
            #time of observations
            disk['time'] = h5f['TIME_DAY_AURORAL'][:] #seconds since start of day
            disk['year'] = h5f['YEAR_DAY_AURORAL'][:] 
            disk['doy'] = h5f['DOY_DAY_AURORAL'][:]

            #read radiances as kR
            disk['radiance_all_colors_uncertainty'] = h5f['DISK_RADIANCE_UNCERTAINTY_DAY_AURORAL'][:] /1000.
            disk['radiance_all_colors'] = h5f['DISK_RECTIFIED_INTENSITY_DAY_AURORAL'][:] / 1000.

            #read in solar zenith angle in degrees
            disk['SZA'] = h5f['PIERCEPOINT_DAY_SZA_AURORAL'][:]

            h5f.close()
        #get epoch from seconds of day, day of year, and year in terms of datetimes 
        dt = np.empty((len(disk['doy']),1),dtype='object')
        for k in range(len(dt)):
            dt[k,0] = datetime.datetime(disk['year'][k],1,1,0,0,0)+datetime.timedelta(days=disk['doy'][k]-1.) + datetime.timedelta(seconds= disk['time'][k])
        disk['epoch'] = dt.flatten()        

        return disk

    def get_ssusi_pass(self):
        """
        Main function to read in and preprocess SDR file. 
        1. Read SDR ncdf file
        2. Convert observations to magnetic coordinates
        3. Applies solar influence removal if desired
        4. Applies Spatial binning if desired

        Returns
        -------
        disk_int : dict
            Dictionary preprocessed observations with elements
            ['epochjd_match'] : np.ndarray (n_obs x 1)
                Array of observation times (in Julian Data) 
            ['observer_ids'] : np.ndarray (n_obs x 1)
                Satellite number associated with observations
            ['Y'] : array_like (n_obs x 1)
                Observations of FUV radiance whose color is specified by radiance_type  (in kilo rayleighs)
            ['Y_var'] : np.ndarray (n_obs x 1)
                Observation error 
            ['mlat'] : np.ndarray (n_obs x 1)
                Magnetic latitude (degrees) of observations in Apex coordinates of reference height 110 km 
            ['mlon'] : np.ndarray (n_obs x 1)
                Magnetic local time of observations (expressed in degrees ) of observations in Apex coordinates 
            ['observer'] : np.ndarrray (n_obs x 1)
                Satellite number associated with observations 
        """
        ssusi_file = self.ssusi_file

        #Step 1: read in each file 
        disk_data = self._read_SDR_file(ssusi_file)
        
        #Step 2: integrate disk data into usable magnetic coordinates
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

        #Step 3: if solar influence removal
        if self.noise_removal:
            radiance_fit = self._radiance_zenith_correction(disk_data['SZA'],disk_int['Y'])                    
            disk_int['Y'] = disk_int['Y'] - radiance_fit
            disk_int['Y'][disk_int['Y']<0] = 0

        #flatten data
        for item in disk_int:
            disk_int[item] = disk_int[item].flatten()

        #get times in terms of jds
        disk_int['epochjd_match'] = datetimearr2jd(disk_int['epoch_match']).flatten()

        #Step 4: spatially bin observations if desired
        if self.spatial_bin:
            disk_int = self.ssusi_spatial_binning(disk_int)

        disk_int['observer'] = np.ones_like(disk_int['epochjd_match']) * self.dmsp

        return disk_int 

    def ssusi_spatial_binning(self,disk_int):
        """
        This function spatially bins the observations using equal solid angle binning.

        Parameters
        ----------
        disk_int : dict
            dictionary from ssusi_pass with elements
            ['epochjd_match'] : np.ndarray (n_obs x 1)
                Array of observation times (in Julian Data) 
            ['observer_ids'] : np.ndarray (n_obs x 1)
                Satellite number associated with observations
            ['Y'] : array_like (n_obs x 1)
                Observations of FUV radiance whose color is specified by radiance_type  (in kilo rayleighs)
            ['Y_var'] : np.ndarray (n_obs x 1)
                Observation error 
            ['mlat'] : np.ndarray (n_obs x 1)
                Magnetic latitude (degrees) of observations in Apex coordinates of reference height 110 km 
            ['mlon'] : np.ndarray (n_obs x 1)
                Magnetic local time of observations (expressed in degrees ) of observations in Apex coordinates 
            ['observer'] : np.ndarrray (n_obs x 1)
                Satellite number associated with observations 
        Returns
        -------
        disk_int : dict
            Same keys as input but binned lol
        """

        disk_binned = {}
        #spatial bin 
        lats_in_pass, lts_in_pass, Y_in_pass, Y_var_in_pass = disk_int['mlat'], disk_int['mlt'], disk_int['Y'], disk_int['Y_var']
        epochjds_in_pass = disk_int['epochjd_match']
        
        #convert from mlt [0, 24] to [-12, 12]
        lts_mask = lts_in_pass >= 12
        lts_in_pass[lts_mask] -= 24 

        #bin observation values
        binlats, binlons, binstats = self.grid.bin_stats(lats_in_pass.flatten(), lts_in_pass.flatten(), Y_in_pass.flatten(), \
                                            statfun = np.nanmean, center_or_edges = 'center')
        #get varaince of each bin 
        binlats, binlons, binstats_var = self.grid.bin_stats(lats_in_pass.flatten(), lts_in_pass.flatten(), Y_var_in_pass.flatten(), \
                                    statfun = np.nanvar, center_or_edges = 'center')
        #bin observation time
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

        Parameters
        ----------
        datadict : dict
            dictionary object from _read_SDR_file()
        Returns
        -------
        alat : np.ndarray (same shape as datadict['glat'])
            Apex latitude
        mlt, : np.ndarray (same shape as datadict['glat'])
            Magnetic local time  
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

    def get_ingest_data(self,startdt = None, enddt = None, hemisphere = None):
        """
        Call to return observations from a polar pass 


        Parameters
        ---------- 
        startdt : datetime object
            Defaults to first available observation time
        enddt : datetime object
            Defaults to last available observation time
        hemisphere : str
            Defaults to hemisphere specified in init 
        
        Returns
        -------
        ylats : np.ndarray
            Absolute magnetic latitudes of observations
        ylons : np.ndarray
            magnetic 'longitudes' (MLT in degrees) of observations
        y : np.ndarray
            1D array of kiloRayleighs
        y_var : np.ndarray
            1D array of uncertainies (variances in kiloRayleighs)
        jds : np.ndarray 

        """
        startdt = jd2datetime(np.nanmin(self['jds'])) if startdt is None else startdt
        enddt = jd2datetime(np.nanmax(self['jds'])) if enddt is None else enddt
        hemisphere = self.hemisphere if hemisphere is None else hemisphere

        datadict = self.get_data_window(startdt,
                                        enddt,
                                        hemisphere,
                                        'all')
        y = datadict['Y'].reshape(-1,1)


        #Format the error/variance vector similarly,
        y_var = datadict['Y_var'].reshape(-1,1);

        #Locations for each vector component
        ylats = datadict['lats'].reshape(-1,1)
        ylons = datadict['lons'].reshape(-1,1)

        #jds 
        jds = datadict['jds'].reshape(-1,1)
        return np.abs(ylats),ylons,y,y_var,jds 

    def plot_obs(self, ax, startdt = None,enddt = None,hemisphere = None, **kwargs):
        """
        Plot observations from a particular polar pass 

        Paramters 
        ---------
        ax : matplotlib axis
        startdt : datetime object
            Defaults to first available observation time
        enddt : datetime object
            Defaults to last available observation time
        hemisphere : str
            Defaults to hemisphere specified in init 
        """
        lats,lons,obs,y_var,jds = self.get_ingest_data(startdt,enddt,hemisphere)
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
            use the output dictionary from the read in function readSSUSISDR()
        """
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
        
        Parameters
        ----------
        SZA - list, num_obsx1
            Solar zenith angle of the radiance observations(degrees)
        radiance - list, num_obsx1
            Radiance observations 
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
        startjd = datetime2jd(startdt)
        endjd = datetime2jd(enddt)

        inrange = np.logical_and(self['jds'].flatten()>=startjd,
                                    self['jds'].flatten()<endjd)
        return inrange

    def _get_hemisphere_mask(self,hemisphere):
        if hemisphere not in ['N','S']:
            return ValueError(('{}'.format(hemisphere)
                              +' is not a valid hemisphere (use N or S)'))
        if hemisphere == 'N':
            inhemi = self['lats'] > self.minlat
        elif hemisphere == 'S':
            inhemi = self['lats'] < -self.minlat
        return inhemi
    def _check_allowed_observers(self,allowed_observers):
        if not isinstance(allowed_observers,list):
            raise RuntimeError('allowed_observers must be a list of '
                               +'DMSP satellite numbers')
        for observer_id in allowed_observers:
            if observer_id not in self.observers:
                raise ValueError('DMSP satellite number {} not'.format(observer_id)
                                 +'in \n({})'.format(self.observers))

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
 