# superpixels

## Setup

- Install conda
- Setup conda env:
  - `conda env create -f environment.yml`
  - `conda activate superpixels`
- Install copick: `pip install git+https://github.com/copick/copick-utils.git`
- Install `cellcanvas_spp` in editable mode:
  - `pip install -e .`
- Environment:
  - `spp_segmentation` requires `COPICK_DATA` to be set
    - e.g., `export COPICK_DATA=COPICK_DATA="/Users/eraymond/Library/CloudStorage/GoogleDrive-eraymond@chanzuckerberg.com/My Drive/2024_CZHackathon_InteractiveImageEmbeddings"`

## Create a config for demo notebook
- USe an existing config OR
- Run setup notebook
  - Edit project_data_dir in setup
