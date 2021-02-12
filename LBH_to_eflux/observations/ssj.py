import numpy as np
import h5py
import glob
import pdb
import os 
import spacepy.pycdf as pycdf
from geospacepy import special_datetime, satplottools

class ssj_day(object):
    """
    This class facilitates the data handling for the SSJ data provided by GLOW model using v2 
    """

    def __init__(self,dmsp, hemi, ssj_file, read_spec = False):
        """
            dmsp - int
                number of the DMSP spacecraft with acceptable numbers being 16, 17, and 18.
            hemi - string
                hemisphere (must be either 'N' or 'S')
            day - datetime object
                datetime element of the day of data to use
            ssj_file- string
                location of ssj file 
        """

        self.dmsp = dmsp
        self.hemi = hemi
        self.ssj_dir = ssj_dir

        self.read_spec = read_spec #determine whether or not to read in spectrogram results

        #read in the data
        self.mlts, self.mlons,self.mlats,self.mean_energy,self.mean_energy_uncert,self.energy_flux,self.energy_flux_uncert, \
        self.lbhl, self.lbhs, self.orbit_index,self.epoch  = ssj_day.read_cdf_ssj()
        
        #convert MLT to degrees 
        self.mlons = self.mlts*15

        #unit conversions 
        #erg 
        # self.energy_flux = self.energy_flux * 1.6e-12 * np.pi
        self.energy_flux_uncert = self.energy_flux_uncert *self.energy_flux

        #convert mean energy unit
        #from eV to KeV
        self.mean_energy = self.mean_energy*1e-3
        self.mean_energy_uncert = self.mean_energy_uncert*self.mean_energy

        #convert epoch to jd time 
        self.epochjd = special_datetime.datetimearr2jd(self.epoch) 

    def read_cdf_ssj(self):
        """
            Read in GLOW processing of SSJ data for the entire day
        """

        day_dir = os.path.join(self.ssj_dir, '{}'.format(self.day.strftime('%Y%m%d')))
        day_ssj =  os.path.join(day_dir,'dmsp-f%d_ssj_precipitating-electrons-ions_%s_v1.1.3_GLOWcond_v2.cdf' % (self.dmsp, (self.day.strftime('%Y%m%d'))))
        self.day_ssj = day_ssj #store this in case we want to read in the spectral data later

        with pycdf.CDF(day_ssj) as cdffile:    
            MLT_dmsp_day =cdffile['SC_APEX_MLT'][:].flatten()
            LON_dmsp_day = cdffile['SC_APEX_LON'][:].flatten()
            LAT_dmsp_day = cdffile['SC_APEX_LAT'][:].flatten()

            #read in precipetation data
            mean_energy_dmsp_day = cdffile['ELE_AVG_ENERGY'][:].flatten()
            mean_energy_dmsp_uncert_day =   cdffile['ELE_AVG_ENERGY_STD'][:].flatten()


            energy_flux_dmsp_day = cdffile['ELE_TOTAL_ENERGY_FLUX'][:].flatten()
            energy_flux_dmsp_uncert_day = cdffile['ELE_TOTAL_ENERGY_FLUX_STD'][:].flatten()

            #read in pseudo obs 
            pseudoLBHL = cdffile['LYMAN_BIRGE_HOPFIELD_LONG'][:]
            pseudoLBHL_solar = cdffile['LYMAN_BIRGE_HOPFIELD_LONG_SOLAR'][:]

            pseudoLBHS = cdffile['LYMAN_BIRGE_HOPFIELD_SHORT'][:]
            pseudoLBHS_solar = cdffile['LYMAN_BIRGE_HOPFIELD_SHORT_SOLAR'][:]

            #read in spectrogram data
            if self.read_spec:
                self.channel_energies = cdffile['CHANNEL_ENERGIES'][:]
                self.ele_diff_energy_flux = cdffile['ELE_DIFF_ENERGY_FLUX'][:]
                self.ion_diff_energy_flux = cdffile['ION_DIFF_ENERGY_FLUX'][:]
                #filter out negative fluxes?
                self.ele_diff_energy_flux[self.ele_diff_energy_flux <= 0 | ~np.isfinite(self.ele_diff_energy_flux) ] = np.nan

                #proton
                self.ion_energy_flux_dmsp = cdffile['ION_TOTAL_ENERGY_FLUX'][:].flatten()

            orbit_index_day = cdffile['ORBIT_INDEX'][:].flatten()
            epoch_dmsp_day = cdffile['Epoch'][:].flatten()

        pseudoLBHL_auroral,  pseudoLBHS_auroral = pseudoLBHL - pseudoLBHL_solar, pseudoLBHS - pseudoLBHS_solar

        return MLT_dmsp_day, LON_dmsp_day,LAT_dmsp_day, mean_energy_dmsp_day, \
                    mean_energy_dmsp_uncert_day, energy_flux_dmsp_day, \
                    energy_flux_dmsp_uncert_day, \
                    pseudoLBHL_auroral, pseudoLBHS_auroral, \
                    orbit_index_day, epoch_dmsp_day
    
    def time_mask(self,startdt,enddt, lat_limit = 50):
        """
            startdt - datetime object
                Start time for time period of interest
            enddt - datetime object
                End time for time period of interest
            lat_limit - float, optional
                absolute lower limit for the magnetic latitude of observations to be used
        """

        #create the hemispheric and mlat mask
        if self.hemi == 'N':
            mask = np.logical_and(self.epoch >=startdt, self.epoch< enddt) & (self.mlats > lat_limit)
        elif self.hemi == 'S':
            mask = np.logical_and(self.epoch >=startdt, self.epoch< enddt) & (self.mlats < -lat_limit)
        else:
            raise ValueError('Hemi needs to be N or S')

        #include a mask by orbit
        orbits_in_time = self.orbit_index[mask]
        first_orbit_in_time = orbits_in_time[0]
        mask = mask & (self.orbit_index == first_orbit_in_time) 

        return mask 

    def time_mask_ssj(self,startdt,enddt, lat_limit = 50,return_rad = False):
        """
            startdt - datetime object
                Start time for time period of interest
            enddt - datetime object
                End time for time period of interest
            lat_limit - float, optional
                absolute lower limit for the magnetic latitude of observations to be used
        """

        #create the hemispheric and mlat mask
        if self.hemi == 'N':
            mask = np.logical_and(self.epoch >=startdt, self.epoch< enddt) & (self.mlats > lat_limit)
        elif self.hemi == 'S':
            mask = np.logical_and(self.epoch >=startdt, self.epoch< enddt) & (self.mlats < -lat_limit)
        else:
            raise ValueError('Hemi needs to be N or S')

        # include a mask by orbit
        orbits_in_time = self.orbit_index[mask]
        first_orbit_in_time = orbits_in_time[0]
        mask = mask & (self.orbit_index == first_orbit_in_time) 

        #apply the mask 
        MLT_dmsp, LON_dmsp, LAT_dmsp, mean_energy_dmsp, energy_flux_dmsp, \
        epoch_dmspjd, mean_energy_dmsp_uncert,energy_flux_dmsp_uncert, lbhl, lbhs  = self.mlts[mask], self.mlons[mask], \
            self.mlats[mask],self.mean_energy[mask], self.energy_flux[mask], self.epochjd[mask], \
            self.mean_energy_uncert[mask],self.energy_flux_uncert[mask], self.lbhl[mask], self.lbhs[mask]


        #check order 
        if return_rad: #return radiance 
            return LON_dmsp.flatten(), LAT_dmsp.flatten(), mean_energy_dmsp.flatten(), energy_flux_dmsp.flatten(), \
                epoch_dmspjd.flatten(), mean_energy_dmsp_uncert.flatten(),energy_flux_dmsp_uncert.flatten(),\
                lbhl, lbhs
        else:
            return LON_dmsp.flatten(), LAT_dmsp.flatten(), mean_energy_dmsp.flatten(), energy_flux_dmsp.flatten(), \
                epoch_dmspjd.flatten(), mean_energy_dmsp_uncert.flatten(),energy_flux_dmsp_uncert.flatten()