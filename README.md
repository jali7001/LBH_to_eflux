# If you ignore me, I'll be angry and disappointed. 

Hi there! 
I've decided to put the code I used to train the neural network model I used for my Master's here.
The code should be broken up into 2 sections/folders.

1. Observations - "observations"
2. Neural Network Training and Analysis  - "nn"

## Observations
### Preprocessing
Here is the preprocessing I used for the SSJ and SSUSI data. I tried to leave most options as toggles and I acknowledge it's probably not the most efficient way of reading the data but it's a good place to start with you ever need to read in these files. 

Right now the SSUSI read file only supports the SSUSI SDR product downloaded from NASA CDAweb (documentation available at https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmspf16/ssusi/documents/ ) and SSJ read file supports NASA CDAweb files and GLOW v.2 processed files (see Liam Kilcommons at CU-Boulder for access to that). 

### Download
If you're feeling lazy, I've also included a download scripts for both SSUSI and SSJ files called .... Download_SSJ.py and Download_SSUSISDR.py in the download sub folder.

### Conjunctions
I've also provided the code I used to find the k-nearest distance weighted averaging I used to get conjunctions between the SSUSI and SSJ observations in SSJ_SSUSI_Conjunction.py.
The script in that file currently assumes the data is organized in the same way as is done by the download scripts. If that's not the case and my code sucks, you should be able to also use the conjunction class functionality in the file.

### Testing
I've also included two jupyter notebooks that plot some stuff from SSUSI and SSJ preprocessed observations. 

## Neural Netowrk 
For the neural network files, I found the visual aid that can be done with jupyter notebooks super helpful. As a result, I left most of these files as .ipynb.
### Training and Feature Engineering

### Analysis 


