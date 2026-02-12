## EM Counterpart of Binary Black Hole Mergers

This repository contains example notebooks to find the EM counterpart properties and the AGN disk properties of a BBH merger.  
Right now, the repository contains the notebooks folder, which includes a couple of example notebooks that explain how calculations are performed and how to import data from GWTC fits files for skymaps and from the SDSS VAC catalogue for the AGNs. The folder also contains the used functions as a class, making it convenient to retrieve the calculated data.  
In the notebooks folder, we have the `BBH_EM_cp_model.ipynb` notebook, which contains all the functions that I have used to calculate the EM properties. I later converted all these functions into a separate class, which I stored in the python file `EM_CP_class.py` (I use these classes defined here in other python/notebook files). `AGN_GW190521.ipynb` is a walkthrough notebook that explains the process of extracting the data from the fits files and running and storing the results from both the AGN disk properties calculating function and the EM counterpart properties calculating function. The `gw190521_em_plots.ipynb` file extracts the results and plots them. Additionally, I have added the plots of the result separately in the plots folder.
All the relevant data are stored in the drive folder below (CAUTION: The pickle files are huge files, approx 5GB and 8GB).
https://drive.google.com/drive/folders/1yPuKjGX_5pIDM2fYjacLvV-2Lr68aYgk  
I am in the process of building this entire repository into a python package that can be directly installed.
