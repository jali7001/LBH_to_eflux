import numpy as np
import h5py
import glob
import pdb
import os 
import spacepy.pycdf as pycdf
from geospacepy import special_datetime, satplottools
from collections import OrderedDict
from geospacepy.special_datetime import (datetimearr2jd,
                                        datetime2jd,
                                        jd2datetime)

class ssj_day(object):
    """
    This class facilitates the data handling for the SSJ data provided by GLOW model using v2 

    Attributes:
    -----------
    ['jds'] : np.ndarray (n_obs x 1)
        Array of observation times (in Julian Date)
    ['epoch'] : datetime list (n_obs x 1)
        Array of observation times in datetime
    ['lats'] : np.ndarray (n_obs x 1)
        Magnetic latitude (degrees) of observations in Apex coordinates of reference height 110 km 
    ['lons'] : np.ndarray (n_obs x 1)
        Magnetic local time of observations (expressed in degrees ) of observations in Apex coordinates 
    ['orbit_index'] : np.ndarray (n_obs x 1)
        orbit number of observation
    ['ele_mean_energy'] : np.ndarray (n_obs x 1)
        Electron mean energy in KeV
    ['ele_mean_energy_uncert'] : np.ndarray (n_obs x 1)
        Electron mean energy uncertainty KeV
    ['ele_total_energy_flux'] : np.ndarray (n_obs x 1)
        Electron total energy flux in ergs/cm^2 s ster
    ['ele_total_energy_flux_uncert'] : np.ndarray (n_obs x 1)
        Electron total energy flux uncertainty in ergs/cm^2 s ster
    ['ion_total_energy_flux'] : np.ndarray (n_obs x 1)
        Ion total energy flux in ergs/cm^2 s ster
    ['ion_total_energy_flux_uncert'] : np.ndarray (n_obs x 1)
        Ion total energy flux uncertainty in ergs/cm^2 s ster
    ['ele_diff_energy_flux'] : np.ndarray (n_obs x 19)
        Electron energy flux across the 19 energy channels ergs/cm^2 s ster.
        Only appears if read_spec = True
    ['ion_diff_energy_flux'] : np.ndarray (n_obs x 19)
        Ion energy flux across the 19 energy channels ergs/cm^2 s ster
        Only appears if read_spec = True
    """
    def __init__(self, dmsp, hemi, ssj_file, read_spec = False, min_lat = 50):
        """
        Parameters
        ----------
            dmsp - int
                 DMSP spacecraft must take value in [16, 17, 18]
            hemi - str
                Hemisphere of observations (must be either 'N' or 'S')
            ssj_file- str
                location of ssj file 
            read_spec - bool, optional
                Switch of whether or not to read differential flux values
                Defaults to False
            min_lat - float, optional
        """

        self.dmsp = dmsp
        self.hemisphere = hemi
        self.ssj_file = ssj_file
        self.read_spec = read_spec #determine whether or not to read in spectrogram results
        self.min_lat = min_lat

        #read in the data
        self.read_cdf_ssj()

        #convert epoch to jd time 
        self['jds'] = special_datetime.datetimearr2jd(self['epoch']) 

    def read_cdf_ssj(self):
        """
        Read in GLOW processing of SSJ data for the entire day.
        """

        ssj_file = self.ssj_file

        with pycdf.CDF(ssj_file) as cdffile:    
            self['lons'] = cdffile['SC_APEX_MLT'][:].flatten() * 15. #report lons at magnetic local time in degrees 
            self['lats'] = cdffile['SC_APEX_LAT'][:].flatten() 

            #read in precipitation data
            self['ele_mean_energy'] = cdffile['ELE_AVG_ENERGY'][:].flatten() * 1e-3 #report as KeV
            self['ele_mean_energy_uncert'] =  self['ele_mean_energy'] * cdffile['ELE_AVG_ENERGY_STD'][:].flatten() #file reports uncertainty as fractional uncertainty so get it in as absolute uncertainty

            self['ele_total_energy_flux'] = cdffile['ELE_TOTAL_ENERGY_FLUX'][:].flatten()
            self['ele_total_energy_flux_uncert'] = self['ele_total_energy_flux'] * cdffile['ELE_TOTAL_ENERGY_FLUX_STD'][:].flatten()

            self['ion_total_energy_flux'] = cdffile['ION_TOTAL_ENERGY_FLUX'][:].flatten()
            self['ion_total_energy_flux_uncert'] = self['ion_total_energy_flux'] * cdffile['ION_TOTAL_ENERGY_FLUX_STD'][:].flatten()


            #read in spectrogram data
            if self.read_spec:
                self.channel_energies = cdffile['CHANNEL_ENERGIES'][:]
                self['ele_diff_energy_flux'] = cdffile['ELE_DIFF_ENERGY_FLUX'][:]
                self['ion_diff_energy_flux'] = cdffile['ION_DIFF_ENERGY_FLUX'][:]

                #filter out negative fluxes 
                self['ele_diff_energy_flux'][self['ele_diff_energy_flux'] <= 0 | ~np.isfinite(self['ele_diff_energy_flux']) ] = np.nan
                self['ion_diff_energy_flux'][self['ion_diff_energy_flux'] <= 0 | ~np.isfinite(self['ion_diff_energy_flux']) ] = np.nan


            self['orbit_index'] = cdffile['ORBIT_INDEX'][:].flatten()
            self['epoch'] = cdffile['Epoch'][:].flatten()

        return 


    def get_ingest_data(self,startdt = None, enddt = None, hemisphere = None):
        """
        Parameters
        ----------
        startdt : datetime object
            Defaults to first available observation time
        enddt : datetime object
            Defaults to last available observation time
        hemisphere : str
            Defaults to hemisphere specified in init 
        
        Returns
        --------
        Subset of observations corresponding to date and hemisphere bounds. 
        """
        lat_limit = self.min_lat

        startdt = jd2datetime(np.nanmin(self['jds'])) if startdt is None else startdt
        enddt = jd2datetime(np.nanmax(self['jds'])) if enddt is None else endt
        hemisphere = self.hemisphere if hemisphere is None else hemisphere

        #create the hemispheric and mlat mask
        if self.hemisphere == 'N':
            mask = np.logical_and(self['epoch'] >=startdt, self['epoch'] < enddt) & (self['lats'] > lat_limit)
        elif self.hemisphere == 'S':
            mask = np.logical_and(self['epoch'] >=startdt, self['epoch'] < enddt) & (self['lats'] < -lat_limit)
        else:
            raise ValueError('Hemi needs to be N or S')

        # include a mask by orbit
        orbits_in_time = self['orbit_index'][mask]
        first_orbit_in_time = orbits_in_time[0]
        mask = mask & (self['orbit_index'] == first_orbit_in_time) 

        data_window = OrderedDict()
        for datavarname,datavararr in self.items():
            data_window[datavarname] = datavararr[mask]

        return data_window 

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
