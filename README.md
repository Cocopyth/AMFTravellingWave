# AMF segmentation and analysis
Full code for paper [Oyarte Galvez, L., Bisot, C., Bourrianne, P. et al. A travelling-wave strategy for plant–fungal trade. Nature 639, 172–180 (2025)](https://www.nature.com/articles/s41586-025-08614-x).

# System requirements

## Python version
Code should run on Python >=3.11.

## Parallel computing
All the code for parallelization requires a HPC SLURM job system. Specifics may need to be worked 
out in the case of large dataset handling. Most of the code will however run without it.


# Setup


## Setup for Linux
Installation should take less than 1 hour.

### Setting up environment

- Install poetry : follow instructions [here](https://python-poetry.org/docs/)

From base folder run:
```
poetry install
```

You can then use the poetry environment to run all the code, notebooks and tests.




## Local.env file

Create a text file named `local.env` in the base folder
(for example: `touch local.env`)

And fill the file with the following lines and adapt them to your situation:

All dropbox related fields (From `APP_KEY` onward) can be filled with arbitrary strings. They are not necessary to the core functionning of the code but filling it is necessary for correct import of dependencies.

```
DATA_PATH=C:\Users\coren\Documents\Code\data_info.json
FIJI_PATH=C:\Users\coren\Documents\Code\fiji-win64\Fiji.app/ImageJ-win64.exe
TEMP_PATH=C:\Users\coren\Documents\Code\temp
STORAGE_PATH=C:\Users\coren\Documents\Code
PASTIS_PATH=C:\Users\coren\Documents\Code/anis_filter/anifilters/bin/ani2D
SLURM_PATH=C:\Users\coren\Documents\Code\temp
SLURM_PATH_transfer=C:\Users\coren\Documents\Code\temp
DROPBOX_PATH = C:\Users\coren\AMOLF-SHIMIZU Dropbox\DATA\PRINCE
PATH_ANALYSIS = C:\Users\coren\AMOLF-SHIMIZU Dropbox\DATA\PRINCE_ANALYSIS
CONDA_PATH = /home/cbisot/miniconda3/etc/profile.d/conda.sh

APP_KEY=
APP_SECRET= 
REFRESH_TOKEN = 
FOLDER_ID=
USER_ID= 

```

To have access to a path: 
Always import from the util.sys

## Data

- If the code is directly pulled from **figshare**, 
data is included and paths should be consistent. 
This includes source data for all main figures as well as a subsample of 
raw data to demonstrate the functioning of the pipeline. The **fighsare** doi is _10.6084/m9.figshare.27889143_

- If the code is cloned from GitHub, 
data must manually be downloaded from figshare and then path needs 
being adapted in `test`, `test_notebooks` and `Figure_plots`(default is inconsistent)

## External tools used 

Those tools are not necessary to run the core part of the pipeline. 
Neither are they necessary to reproduce main figures.
These can be ignored in a first installation.

### Fiji:
- This is only necessary for image stitching (first step of the pipeline)
- Chose a location on the computer and download:
https://imagej.net/software/fiji/downloads


### anisotropic filtering:

- This is only necessary for "optimal" skeleton image segmentation
- Chose a location on the computer and download:
http://forge.cbp.ens-lyon.fr/redmine/projects/anifilters

# Reproducibility

## Generating  Figures

- Python notebooks generating Fig. 2, 3 and 5 can be found
in `Figure_plots` folder. It uses source data that is directly incorporated in the figshare repo.

- Figure 4 can be generated using the Matlab script that can be found in the `Figure_plots/Fig4` folder.

- All the External data figures can be reproduced with the corresponding notebooks provided in `Figure_plots` folder.

## Reproducing the pipeline

- All the steps of the pipeline can be reproduced on the example dataset provided
by running. `poetry run python -m unittest .\test\pipeline\test_image_analysis.py`
- All the post-processing steps that generate the datatables can be reproduced 
with `poetry run python -m unittest .\test\pipeline\test_post_processing.py`

## Flow analysis

- All the code related to flow analysis can be found in `flow_analysis`. 
Readme tutorials are provided to reproduce the main steps. 

## Network efficiency and BAS extraction
- This code is made in matlab, scripts and readme are provided in `Figure_plots/Fig4` folder

## DAPI staining nuclei analysis

- All the code related to extracting nuclear density from images can be found 
in `flow_analysis`. 
Readme tutorials are provided to reproduce the main steps.
## Demo notebooks
Example of code showing network manipulation is provided in `test/test_network.ipynb`. It requires 
 test data that can be downloaded [here](https://figshare.com/articles/dataset/Untitled_Item/23902032) to run.
(pulling the global figshare repo should otherwise suffice)
It should plot a small AMF network and display typical methods
to access graph nodes and edges. It should run in less than 2 minutes. 


# Presentation of the `amftrack` repository
## ml
Contains code related to width extraction that is not relevant 
to this article but which dependencies are necessary for proper import statements.

## pipeline
### control
Contains a notebook that displays the typical workflow 
for analysis of a plate.
Dropbox linking and supercomputing facilities are necessary
for proper functioning.
### functions
Contains all functions used for processing of fungal network
graphs (`image_processing`) as well as functions used for 
datatable creation that contains plate and hypha level observables (`post_processing`).

### launching
Contains necessary code for running massive dataset on 
Dutch supercomputer Snellius. Script launcher show typical parameters
used for the analysis of growing AMF fungal networks.

### scripts
Contains the scripts launched sequentially by the script launchers.
These scripts make use of the functions defined in `functions`

## transfer
Contains the script necessary for transferring data to and from dropbox.
This is specific to the data management of the project but some 
dependencies are necessary for proper functioning of import statements.

## util
Contains useful functions to the whole repository.



## test




### Launching tests
Tests can be launched with the following command:
```
poetry run python -m unittest discover . "test*.py"
```

Runing only one test:
```
poetry run python -m unittest -v ~/Wks/AMFtrack/test/util/test_geometry.py
```

Test can also be run with `pytest` if installed (prettier display)
```bash
pytest test_file.py -k test_function
```

### Special tests
For some tests, a processed plate is required. This is directly included when pulling the
whole package from figshare.
Otherwise, such test data can be downloaded at [this link](https://figshare.com/articles/dataset/Untitled_Item/23902032) .
The data folders must be stored at the path `path_data`.
If the data is not present, the tests will be skipped.
The tests can be safely run even if no `path_data` directory is present.

Some tests create and save plots in the **test** directory.
These files don't accumulate (they are just replace at each test).


##Note on coordinates

The general choice of coordinates is:
x -> for the small vertical axis (3000 for prince images)
y -> for the long horizontal axis

This choice is consistent accross `general`, `timestep` and `image` referential in the `exp` object.
As a result:
- we write coordinates as `[x, y]`
- np.arrays have the shape (dim_x, dim_y) and can be shown with plt.imshow()
- to access a coordinate in an image we use `im[x][y]`

# Model simulations

Notebooks generating the figure for model simulations can be found
at https://github.com/Cocopyth/model_integration. 

It requires installing dolfin 2019.2.0.dev0 which cannot easily be
done within a poetry environment. 


# Future development

This repository is a "freezed" version of code developed for mycorrhizal network analysis 
at the time of publication and is meant for reproducibility.
For researcher interested in using similar technique for their research, 
I advise the following:

- For everything related to **segmentation** of "filaments", depending on imaging specificities, this code may not be
readily used. One can explore the use of frangi filters, bowler hat filters and anisotropic filtering and maybe reuse
some of the functions of this repository.
- For everything related to **tracking** of hyphae, the method developped for this paper is analogous to tracking as it
can be done with Spatio-temporal hypergraph. I advise to consult the following
[library](https://github.com/amin942001/STHype/tree/hypergraph_class) where demo notebooks provide a good overview.
- The **flow analysis** is currently being implemented in python. First releases will be posted here.
