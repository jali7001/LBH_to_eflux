from ftplib import FTP_TLS
import datetime,os
from pathlib import Path


"""
This script downloads the SSSUI SDR data for the time period used for my thesis. 
Feel free to change the dates for observations of your choice here.
The script will FTP NASA cdaweb and store all of the SSUSI cdfs for each day in one folder inside the given dir.
"""

year = 2014 #years to download data for
doy_arr = [48,49,50,51,52,53,54] #days of year 
satNum_arr = [16,17,18] #DMSP satellites to take data from


def download_all(ftp,full_remote_dir,local_dir):
    ftp.cwd(full_remote_dir)
    filenames = ftp.nlst() # get filenames within the directory
    print('Downloading from %s' % (full_remote_dir))
    print(filenames)

    for filename in filenames:
        print('Downloading %s...' % (filename))
        local_filename = os.path.join(local_dir, filename)
        f = open(local_filename, 'wb')
        ftp.retrbinary('RETR '+ filename, f.write)
        f.close()


directory = input("Enter the directory where you would like to store SSUSI SDR data. If default location, press ENTER: ")

#if default dir, store in parent directory (should be repo)
if directory == '':
    print("No directory specified, placing data in repository folder")
    current_path = os.path.dirname(os.path.realpath(__file__))
    directory = os.path.join(Path(current_path).parent, 'SSUSI_SDR_Observations')

#if path does not exist, make it 
if not os.path.exists(directory):
    os.makedirs(directory)

#iterate across day of year 
for doy in doy_arr:
    day_dir = os.path.join(directory,'{}{}'.format(year,doy))
    if not os.path.exists(day_dir):
        os.makedirs(day_dir)

    #iterate across satellites
    for satnum in satNum_arr:
        ssusi_dir_sdr_disk = '/pub/data/dmsp/dmspf%d/ssusi/data/sdr-disk/%d/%.3d' % (satnum,year,doy) #create query for disk observations
        ftp = FTP_TLS('cdaweb.gsfc.nasa.gov')
        ftp.login()
        download_all(ftp,ssusi_dir_sdr_disk, local_dir = day_dir)

ftp.quit()