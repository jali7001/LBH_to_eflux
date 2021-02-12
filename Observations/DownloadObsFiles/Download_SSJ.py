from ftplib import FTP_TLS
import datetime,os
from pathlib import Path
import re

"""
This script downloads the SSJ data for the time period used for my thesis. 
Feel free to change the dates for observations of your choice here.
The script will FTP NASA cdaweb and store all of the SSUSI cdfs for each day in one folder inside the given dir.
"""

year = 2014 #years to download data for
doy_arr = [48,49,50,51,52,53,54] #days of year 
satNum_arr = [16,17,18] #DMSP satellites to take data from


def download_ssj_dayfile(ftp,full_remote_dir,local_dir, datetimestr,satnum):
    """
    Function to download SSJ file from NASA CDAweb
    """
    ftp.cwd(full_remote_dir)
    filenames = ftp.nlst() # get filenames within the directory

    #get the file that matches the date
    matching = [s for s in filenames if datetimestr in s]

    if len(matching) == 0:
        print('No files found for {} F{}'.format(datetimestr, satnum))
        return

    print('Downloading from %s' % (full_remote_dir))
    print(matching)
    
    for filename in matching:
        print('Downloading %s...' % (filename))
        local_filename = os.path.join(local_dir, filename)
        f = open(local_filename, 'wb')
        ftp.retrbinary('RETR '+ filename, f.write)
        f.close()
    return


directory = input("Enter the directory where you would like to store SSJ data. If default location, press ENTER: ")

#if default dir, store in parent directory (should be repo)
if directory == '':
    print("No directory specified, placing data in repository folder")
    current_path = os.path.dirname(os.path.realpath(__file__))
    directory = os.path.join(Path(current_path).parent, 'SSJ_Observations')

#if path does not exist, make it 
if not os.path.exists(directory):
    os.makedirs(directory)

for doy in doy_arr:
    day_dir = os.path.join(directory, '%d%d' % (year,doy))
    if not os.path.exists(day_dir):
        os.makedirs(day_dir)
    for satnum in satNum_arr:
        ssj_electron_ion = '/pub/data/dmsp/dmspf%d/ssj/precipitating-electrons-ions/%d/' % (satnum,year)
        dt_str =  (datetime.datetime(year,1,1) + datetime.timedelta(doy - 1)).strftime('%Y%m%d')

        ftp = FTP_TLS('cdaweb.gsfc.nasa.gov')
        ftp.login()
        download_ssj_dayfile(ftp,ssj_electron_ion, day_dir,dt_str,satnum)
ftp.quit()