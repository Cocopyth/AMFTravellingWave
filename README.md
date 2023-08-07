# AMF segmentation and analysis
AMF segmentation


# Setup

[//]: # (## Setup with conda)

[//]: # (*For the script*)

[//]: # ()
[//]: # (```bash)

[//]: # (conda install -c open3d-admin open3d==0.9.0)

[//]: # (conda install -c anaconda scipy)

[//]: # (conda install -c anaconda pandas)

[//]: # (conda install -c anaconda networkx)

[//]: # (conda install -c conda-forge matplotlib)

[//]: # (pip install pymatreader)

[//]: # (conda install -c anaconda numpy)

[//]: # (conda install -c conda-forge opencv)

[//]: # (pip install imageio #use pip here to avoid conflict)

[//]: # (conda install -c conda-forge jupyterlab)

[//]: # (pip install pycpd)

[//]: # (pip install cython)

[//]: # (git clone https://github.com/gattia/cycpd)

[//]: # (cd cycpd)

[//]: # (sudo python setup.py install)

[//]: # (pip install bresenham)

[//]: # (conda install scikit-image)

[//]: # (conda install -c conda-forge scikit-learn )

[//]: # (pip install Shapely)

[//]: # (pip install tqdm)

[//]: # (pip install dropbox)

[//]: # (pip install imageio-ffmpeg)

[//]: # (```)

[//]: # (<!-- - conda install -c anaconda ipykernel -->)

[//]: # ()
[//]: # (*For nice display*)

[//]: # (```bash)

[//]: # (conda install -c conda-forge ipympl)

[//]: # (conda install -c conda-forge nodejs)

[//]: # (conda install -c conda-forge/label/gcc7 nodejs)

[//]: # (conda install -c conda-forge/label/cf201901 nodejs)

[//]: # (conda install -c conda-forge/label/cf202003 nodejs)

[//]: # (jupyter labextension install @jupyter-widgets/jupyterlab-manager)

[//]: # (jupyter lab build)

[//]: # (```)

useful jupyterlab extensions:
https://github.com/jpmorganchase/jupyter-fs

## Setup for Linux

### Setting up environment

From base folder:
```
virtualenv --python=python3 venv
```
(or replace `python3` by the path to the python version you want to clone.)

Launching environment:

```
source venv/bin/activate
```

### Install requiered packages

Activate the environnement before launching

`pip3 install -r requirements.txt`

Additionnal packages to install:

```bash
git clone https://github.com/gattia/cycpd
cd cycpd
sudo python setup.py install
```

For github authentification 

conda install gh --channel conda-forge 

Install Fiji:

Chose a location on the computer and download:
https://imagej.net/software/fiji/downloads

Install anisotropic filtering:

Chose a location on the computer and download:
http://forge.cbp.ens-lyon.fr/redmine/projects/anifilters

### Install the package in editable mode

For better display:

`jupyter labextension install @jupyter-widgets/jupyterlab-manager`
`jupyter lab build`

### Install the package in editable mode
Remove the *pyproject.toml* file (for poetry)

To run from the base folder:
(will run the setup.py script)
`pip install -e .`
###Install the font for plotting purposes
Copy font in your environment fonts. Example on snellius "/gpfs/home2/cbisot/miniconda3/envs/amftrack/lib/python3.7
/site-packages/matplotlib/mpl-data/fonts/ttf/lucidasansdemibold.ttf"

The code should be modified in order to avoid failure in case the font is not present.

### Local.env file

Create a text file named `local.env` in the base folder
(for example: `touch local.env`)

And fill the file with the following lines and adapt them to your situation:

All dropbox related fields can be filled with arbitrary strings. They are not necessary to the core functionning of the code but filling it is necessary for correct import of dependencies.

```
DATA_PATH=C:\Users\coren\Documents\PhD\Code\data_info.json
FIJI_PATH=C:\Users\coren\Documents\PhD\Code\fiji-win64\Fiji.app/ImageJ-win64.exe
TEMP_PATH=C:\Users\coren\Documents\PhD\Code\temp
STORAGE_PATH=C:\Users\coren\Documents\PhD\Code
PASTIS_PATH=/home/ipausers/bisot/anis_filter/anifilters/bin/ani2D
SLURM_PATH=/data/temp
SLURM_PATH_transfer=/data/temp
DROPBOX_PATH = C:\Users\coren\AMOLF-SHIMIZU Dropbox\DATA\PRINCE
DROPBOX_PATH_ANALYSIS = C:\Users\coren\AMOLF-SHIMIZU Dropbox\DATA\PRINCE_ANALYSIS
CONDA_PATH = /home/cbisot/miniconda3/etc/profile.d/conda.sh

APP_KEY=
APP_SECRET= 
REFRESH_TOKEN = 
FOLDER_ID=
USER_ID= 

```

To have access to a path: 
Always import from the util.sys

# Presentation of the repository

##p
## Tests

### 1/ Generality
The tests are all in the `test` folder.
The python module chosen for tests is `unittest`.
https://docs.python.org/3/library/unittest.html

All test files must start with `test`. All test function and classes must start with `test`.

**Ex**: `test_sys_util.py`

And all testing classes must be subclass from the unittest base test class and must start with Test.

The file **helper.py** contains utils for testing: mock object, skipping functions, ..

### 2/ Launching tests
Tests can be launched with the following command:
```
python3 -m unittest discover . "test*.py"
```

Runing only one test:
```
python3 -m unittest -v ~/Wks/AMFtrack/test/util/test_geometry.py
```

Test can also be run with `pytest` if installed (prettier display)
```bash
pytest test_file.py -k test_function
```

### 3/ Special tests
For some tests, a processed Prince plate is required. Or other types of files.
The data file must be stored at the following path:
**storage_path** + "**test**".
If the data is not present, the tests will be skipped.
The tests can be safely run even if to test/ directory is present.

Some tests create and save plots in the **test** directory.
These files don't accumulate (they are just replace at each test).



## Coordinates

The general choice of coordinates is:
x -> for the small vertical axis (3000 for prince images)
y -> for the long horizontal axis

This choice is consistent accross `general`, `timestep` and `image` referential in the `exp` object.
As a result:
- we write coordinates as `[x, y]`
- np.arrays have the shape (dim_x, dim_y) and can be shown with plt.imshow()
- to access a coordinate in an image we use `im[x][y]`