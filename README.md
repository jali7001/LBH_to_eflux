# If you ignore me, I'll be angry and disappointed. 

Hi there! 
I've decided to put the code I used to train the neural network model I used for my Master's here. 

# Setup
Follow these instructions if you want this repo to behave like the good little package it is.  

1. Clone the repo.
```{sh}
git clone https://github.com/jali7001/LBH_to_eflux.git
``` 
2. Get the dependencies
```{sh}
cd LBH_to_eflux
pip install -r requirements.txt
```
3. Install the package. You should probably pick develop mode if you want to make changes to the scripts.
```{sh}
python setup.py develop
````

# Content
I broke up the code into 2 sections/folders. Really these could be separate repositories but I got a kick out of seeing my thesis code in one nice bundle. 

1. Observations - "observations"
2. Neural Network Training and Analysis  - "nn"

## Observations
### Preprocessing
Here is the preprocessing I used for the SSJ and SSUSI data. I tried to leave most options as toggles and I acknowledge it's probably not the most efficient way of reading the data but it's a good place to start with you ever need to read in these files. 

Right now the SSUSI read file only supports the SSUSI SDR product downloaded from NASA CDAweb (documentation available at https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmspf16/ssusi/documents/ ) and SSJ read file supports NASA CDAweb files and GLOW v.2 processed files (see Liam Kilcommons at CU-Boulder for access to the latter). 

### Download
If you're feeling lazy, I've also included a download scripts for both SSUSI and SSJ files called .... Download_SSJ.py and Download_SSUSISDR.py in the download sub folder.

### Conjunctions
I've also provided the code I used to find the k-nearest distance weighted averaging I used to get conjunctions between the SSUSI and SSJ observations in SSJ_SSUSI_Conjunction.py.
The script in that file currently assumes the data is organized in the same way as is done by the download scripts. If that's not the case and my code sucks, you should be able to also use the conjunction class functionality in the file.

### Testing
I've also included two jupyter notebooks that plot some stuff from SSUSI and SSJ preprocessed observations. This would also be a good place to look at if you want to understand the API to interact with the observation files. 

## Neural Network 
For the neural network files, I found the visual aid that can be done with jupyter notebooks super helpful. As a result, I left most of these files as .ipynb.
### Training and Feature Engineering
training_neural_network.ipynb walks through how I used the conjunction files between SSJ and SSUSI observations to train my neural network model. Most of the feature engineering is pretty standard and the model is fairly simple for deep learning purposes, but I still think it's a good place to start if you ever want to make the model more complicated.

### Analysis 
neural_network_predictions.ipynb is the notebook I used to judge my neural network iterations for my thesis. 


