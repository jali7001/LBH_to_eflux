import sklearn.neighbors
from numpy import linalg as LA
from apexpy import Apex
import numpy as np

#Create an Apex conversion instance at the usual reference altitude
#no epoch is specified; we will set the epoch just-in-time when we are going to
#do an coordinate transformation
apex_reference_height = 110000. # Apex reference height in meters
module_Apex = Apex(refh=apex_reference_height/1000.)

def update_apex_epoch(dt):
    year = dt.year
    doy = dt.timetuple().tm_yday
    epoch = year+doy/(366. if np.mod(year,4)==0 else 365.)
    print('Setting Apex epoch for {} to {}'.format(dt.strftime('%Y%m%d'),epoch))
    module_Apex.set_epoch(epoch)


def dmsp_map_interpolate_NN_smooth_great_circle(lat_dmsp, lon_dmsp, lat_map, lon_map, Obs_map, k = 5, tol = 1.5):
    """
    generic function to spatially interpolate with the SSJ data using nearest neighbors using some arbirtary distance tolerance
    """
    tol = np.deg2rad(tol)
    #reshape to N by 2 array where each row is (lat, lon)
    dmsp_points = np.deg2rad(np.hstack((lat_dmsp.flatten().reshape(-1,1),lon_dmsp.flatten().reshape(-1,1))))
    map_points = np.deg2rad(np.hstack((lat_map.flatten().reshape(-1,1), lon_map.flatten().reshape(-1,1))))
    N_points = dmsp_points.shape[0]
    obs_val = Obs_map.flatten()
    
    model = sklearn.neighbors.NearestNeighbors(n_neighbors = k, radius = tol, metric = 'haversine')

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

def latlt2polar(lat,lt,hemisphere):
    """
    Converts an array of latitude and lt points to polar for a top-down dialplot (latitude in degrees, LT in hours)
    i.e. makes latitude the radial quantity and MLT the azimuthal 

    get the radial displacement (referenced to down from northern pole if we want to do a top down on the north, 
    or up from south pole if visa-versa)
    """
    from numpy import pi
    if hemisphere=='N':
        r = 90.-lat
    elif hemisphere=='S':
        r = 90.-(-1*lat)
    else:
        raise ValueError('%s is not a valid hemisphere, N or S, please!' % (hemisphere))
    #convert lt to theta (azimuthal angle) in radians
    theta = lt/24. * 2*pi

    #the pi/2 rotates the coordinate system from
    #theta=0 at negative y-axis (local time) to
    #theta=0 at positive x axis (traditional polar coordinates)
    return r,theta
def polar2dial(ax):
    """
    Turns a matplotlib axes polar plot into a dial plot
    """
    #Rotate the plot so that noon is at the top and midnight
    #is at the bottom, and fix the labels so radial direction
    #is latitude and azimuthal direction is local time in hours
    ax.set_theta_zero_location('S')
    theta_label_values = np.array([0.,3.,6.,9.,12.,15.,18.,21.])*180./12
    theta_labels = ['%d:00' % (int(th/180.*12)) for th in theta_label_values.flatten().tolist()]
    ax.set_thetagrids(theta_label_values,labels=theta_labels)

    r_label_values = 90.-np.array([80.,70.,60.,50.,40.])
    r_labels = [r'$%d^{o}$' % (int(90.-rv)) for rv in r_label_values.flatten().tolist()]
    ax.set_rgrids(r_label_values,labels=r_labels)
    ax.set_rlim([0.,40.])
def map_polar2cart(LAT,LON, hemi = 'N'):
    #convert latitude and longitude (in degrees) to cartesian coordinates for interpolation purposes
    X_map, Y_map = satplottools.latlon2cart(LAT.flatten(), LON.flatten(),hemi)
    return X_map, Y_map
def dmsp_map_interpolate(X_dmsp, Y_dmsp, X_map, Y_map, tolerance = 0.5):
    """
    generic function to spatially interpolate with the SSJ data using nearest neighbors using some arbirtary distance tolerance
    """

    #indices of the map that fit the dmsp map
    indices = scipy.interpolate.griddata((X_map,Y_map), np.arange(len(X_map.flatten())), (X_dmsp,Y_dmsp), method = 'nearest')

    #get mask for map elements that are within distance tolerance 
    mask = (abs(X_map[indices] - X_dmsp) < tolerance) & (abs(Y_map[indices] - Y_dmsp) < tolerance)

    return indices,mask 
def greatCircleDist(location1,location2,lonorlt='lt'):
    #Returns n angular distances in radians between n-by-2 numpy arrays
    #location1, location2 (calculated row-wise so diff between 
    #location1[0,] and location2[0,]
    #assuming that these arrays have the columns lat[deg],localtime[hours] 
    #and that they are points on a sphere of constant radius
    #(the points are at the same altitude)
    pi = np.pi
    azi2rad = pi/12. if lonorlt=='lt' else pi/180
    wrappt = 24. if lonorlt=='lt' else 360.
    #Bounds check
    over = location1[:,1] > wrappt
    under = location1[:,1] < 0.
    location1[over,1]=location1[over,1]-wrappt
    location1[under,1]=location1[under,1]+wrappt

    if location1.ndim == 1 or location2.ndim == 1:    
        dphi = abs(location2[1]-location1[1])*azi2rad
        a = (90-location1[0])/360*2*pi #get the colatitude in radians
        b = (90-location2[0])/360*2*pi
        C =  np.pi - np.abs(dphi - np.pi)#get the angular distance in longitude in radians
    else:
        dphi = abs(location2[:,1]-location1[:,1])*azi2rad
        a = (90-location1[:,0])/360*2*pi #get the colatitude in radians
        b = (90-location2[:,0])/360*2*pi
        C =  np.pi - np.abs(dphi - np.pi)#get the angular distance in longitude in radians
    return arccos(cos(a)*cos(b)+sin(a)*sin(b)*cos(C))

def myGreatCircleDistance(location1,location2):
    #add a dimension
    location1 = location1.reshape(1, 2)
    location2 = location2.reshape(1, 2)

#     location2.shape = (1,)+location2.shape[:,1]
    angular_distance = greatCircleDist(location1,location2,lonorlt='lon')
    return angular_distance

def dmsp_map_interpolate_NN_smooth(X_dmsp, Y_dmsp, X_map, Y_map, Obs_map, k = 5, tol = 3):
    """
    generic function to spatially interpolate with the SSJ data using nearest neighbors using some arbirtary distance tolerance
    """
    #reshape to N by 2 array where each row is (X, Y)
    dmsp_points = np.hstack((X_dmsp.flatten().reshape(-1,1),Y_dmsp.flatten().reshape(-1,1)))
    map_points = np.hstack((X_map.flatten().reshape(-1,1), Y_map.flatten().reshape(-1,1)))
    N_points = dmsp_points.shape[0]
    obs_val = Obs_map.flatten()
    model = sklearn.neighbors.BallTree(map_points,leaf_size = 40 )
    dists, inds = model.query(dmsp_points, k=k) 

    obs_interp = np.empty(N_points)
    for i in range(N_points):
        norm = LA.norm(dists[i])
        if (norm > tol):
            obs_interp[i] = np.nan
        else:
#             weights = dists[i]/norm

            weights = dists[i]/np.nansum(dists[i])
            obs_interp[i] = np.nansum( obs_val[inds[i]] * weights )

    return obs_interp

def dmsp_map_interpolate_NN_smooth_great_circle(lat_dmsp, lon_dmsp, lat_map, lon_map, Obs_map, k = 5, tol = 1.5):
    """
    generic function to spatially interpolate with the SSJ data using nearest neighbors using some arbirtary distance tolerance
    """
    
    tol = np.deg2rad(tol)
    #reshape to N by 2 array where each row is (lat, lon)
    dmsp_points = np.deg2rad(np.hstack((lat_dmsp.flatten().reshape(-1,1),lon_dmsp.flatten().reshape(-1,1))))
    map_points = np.deg2rad(np.hstack((lat_map.flatten().reshape(-1,1), lon_map.flatten().reshape(-1,1))))
    N_points = dmsp_points.shape[0]
    obs_val = Obs_map.flatten()
    model = sklearn.neighbors.NearestNeighbors(n_neighbors = k, radius = tol, metric = 'haversine')

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

from ssj_auroral_boundary import dmsp_spectrogram
def jd2dayhour(jds):
    #assume jd is an array
    temp = jds - 0.5
    hours = (temp - np.floor(temp))*24 
    return hours
