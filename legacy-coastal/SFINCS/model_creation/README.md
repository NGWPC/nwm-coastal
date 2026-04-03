# Overview
This directory contains Python scripts to create SFINCS model files for a given polygon in one of the 4 NWM v3 coastal domains, that is, hawaii, prvi, pacific and atlgulf.


# Prerequisite
First, the bathymetry data for the NWM v3 domains are needed. The bathmetry data is specified in the HydroMT datacatlog file in YAML format. Here is an example of a datacatalog file.

```yaml
>meta:
>  version: '1'
>  root: ../../hydromt_sfincs/examples/tmpdir
>hawaii:
>  crs: 4326
>  data_type: RasterDataset
>  driver: raster
>  meta:
>    category: topography
>    source_url: s3://edfs-data/surface/nws-topobathy 
>    unit: m+EGM2008
>  version: 2010
>  path: hawaii_30m.tif
```

Second, a geojson polygon file that defines the modeling area is needed.
Third, a geojson polygon file that defines the seaward boundary is needed to define the waterlevel boundary.
Third, two SCHISM parameter files, hgrid.nc and nwmReaches.csv, are needed to create the sfincs.src file.

# Script description
makeOcean_fvcom.py : 
sfincs_model_setup.py : The main driver of the Python scripts. Includes the top level logic for creating a SFINCS model files for a given area.
SCHISMNcGrid.py : The class to manage a SCHISM grid, hgrid.nc, file.
SCHISMnwmReaches : The class to manage a SCHISM to NWM domain crosswalk file.


# Script Usage
The driver script has a help option (-h) for printing usage information.

```python
>usage: sfincs_model_setup.py [-h]
>                             data_catalog model_root_dir polygon_file nwm_domain resolution
>                             waterlevel_boundary_file hgrid_file nwmreach_file
>
>positional arguments:
>  data_catalog          the hydromt data catalog filename
>  model_root_dir        the directory where model files will be saved
>  polygon_file          the jeojson file defines the simulation domain
>  nwm_domain            the nwm domain name, one of atlgulf, pacific, prvi and hawaii
>  resolution            defind grid resolution in meters
>  waterlevel_boundary_file
>                        the jeoson file defines the waterlevel boundary
>  hgrid_file            SCHISM hgrid.nc file
>  nwmreach_file         SCHISM nwmReaches.csv file
>
>options:
>  -h, --help            show this help message and exit
```
#### Examples ####
```bash
./sfincs_model_setup.py ./data_catalog_atlgulf.yml sfincs_atlgulf /shareds3/zhengtao/domain_polygon_atl3.geojson "atlgulf" 200 /shareds3/zhengtao/boundary_atl2.geojson  /efs/ngwpc-coastal/parm/coastal/atlgulf/hgrid.nc /efs/ngwpc-coastal/parm/coastal/atlgulf/nwmReaches.csv
```
