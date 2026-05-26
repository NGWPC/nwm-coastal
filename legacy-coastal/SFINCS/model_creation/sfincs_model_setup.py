#!/usr/bin/env python

# Step 0: Imports
import os
import logging
import argparse
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS

from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils

from SCHISMNcGrid import SCHISMNcGrid
from SCHISMnwmReaches import SCHISMnwmReaches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_catalog', type=str, help='the hydromt data catalog filename')
    parser.add_argument('model_root_dir', type=str, help='the directory where model files will be saved')
    parser.add_argument('polygon_file', type=str, help='the jeojson file defines the simulation domain')
    parser.add_argument('nwm_domain', type=str, help='the nwm domain name, one of atlgulf, pacific, prvi and hawaii')
    parser.add_argument('resolution', type=str, help='defind grid resolution in meters', default=200)
    parser.add_argument('waterlevel_boundary_file', type=str, help='the jeoson file defines the waterlevel boundary')
    parser.add_argument('hgrid_file', type=str, help='SCHISM hgrid.nc file')
    parser.add_argument('nwmreach_file', type=str, help='SCHISM nwmReaches.csv file')

    args = parser.parse_args()

    # Step 1: Initialize the SfincsModel class with the data catalog, the "w+" option overwrites an existing model in the root directory if it exists
    sf = SfincsModel(data_libs=[args.data_catalog], root=args.model_root_dir, mode="w+")

    # Step 2: Use the pre-defined polygon geojson to define the grid

    #HydorMT doesn't support crs from a string, only support EPSG codes
    #lcc = CRS.from_proj4(proj4_str)
    #lcc = CRS.from_user_input(esri_pe_string)

    epsg_code = None
    if ( args.nwm_domain == 'hawaii' ):
        epsg_code = 32604 #UTM
    elif ( args.nwm_domain == 'prvi' ):
        epsg_code = 3920  #UTM
    elif ( args.nwm_domain == 'atlgulf' or args.nwm_domain == 'pacific' ):
        epsg_code = 5070
    else:
        raise ValueError(f"Unknown nwm_domain name - {args.nwm_domain}")

    sf.setup_grid_from_region(
      region={'geom': args.polygon_file}, # we will use the open_boundary from SCHISM, which outlines this particular domain 
      res=float(args.resolution), # resolution in meters - set to be fairly coarse for initial testing
      rotated=False, # do not rotate
      crs=epsg_code, # crs/ epsg code
    )

    # Display the automatically generated input file
    print(sf.config)

    da = sf.data_catalog.get_rasterdataset(args.nwm_domain, geom=sf.region, buffer=5)

    # show the model grid outline
    _ = sf.plot_basemap(fn_out=f"model_grid.png", plot_region=True, bmap="sat", zoomlevel = 12)

    # Step 3: Load elevation dataset(s) and map them to the model grid

    datasets_dep = [
    #{"elevtn":"<highest priority dataset here>"}, # optional zmin, zmax let's you determine what part of data you want
    #{"elevtn":"<next highest priority to fill in where missing, etc>"}, # the dataset order is important
    {"elevtn":args.nwm_domain},
    #{}
    ]

    dep = sf.setup_dep(datasets_dep=datasets_dep)

    # plot
    _ = sf.plot_basemap(fn_out="f{args.nwm_domain}_elev.png", variable='dep',plot_region=True, bmap="sat", zoomlevel=12)

    # Step 4a: Make the mask by setting inactive cells (=0) and active cells (=1)

    sf.setup_mask_active(include_mask=args.polygon_file, reset_mask=True)

    # plot
    _ = sf.plot_basemap(fn_out=f"{args.nwm_domain}_mask.png", variable="msk",plot_region=False,bmap="sat",zoomlevel=12)

    # Step 4b (optional): exclude SCHISM land boundary areas (set inactive (=0))

    # These are messing up the water level boundary part - skipping this step

    # These are the land boundaries from SCHISM - the user may wish to change the inland extents to something different
    # exclude_files = [
    #     'hi_boundaries\land_boundary_0.geojson',
    #     'hi_boundaries\land_boundary_1.geojson',
    #     'hi_boundaries\land_boundary_2.geojson',
    #     'hi_boundaries\land_boundary_3.geojson',
    #     'hi_boundaries\land_boundary_4.geojson',
    # ]

    # gdf_exclude = gdf_exclude = gpd.GeoDataFrame(
    #     pd.concat([gpd.read_file(f) for f in exclude_files], ignore_index=True),
    #     crs=gpd.read_file(exclude_files[0]).crs,
    # )

    # sf.setup_mask_active(exclude_mask=gdf_exclude,reset_mask=False)

    # _ = sf.plot_basemap(variable="msk",plot_region=False,bmap="sat",zoomlevel=12)

    # Step 5: Set the mask open water level boundary (=2)

    sf.setup_mask_bounds(btype="waterlevel",include_mask=args.waterlevel_boundary_file, reset_bounds=True)

    # plot
    _ = sf.plot_basemap(fn_out=f"{args.nwm_domain}_make_wl_bnd.png", variable="msk",plot_region=False,bmap="sat",zoomlevel=12)

    # Step 6: Add river inflow points

    # we will need to connect to NWM/NGen here, will skip this step for now
    # Step 7: Spatially varying roughness data

    # do we have access to a roughness dataset? or a land use dataset that we can convert to roughness? Will skip this step for now and set Manning in step 8.
    # Step 8: Make Subgrid Tables

    sf.setup_subgrid(
        datasets_dep = datasets_dep,
        nr_subgrid_pixels = 20,
        manning_land = 0.04,
        manning_sea = 0.02,
        write_dep_tif = True,
    )

    # Step 9: Add spatially varying infiltration data

    # Will skip this step, but option to include it is available
    # Step 10: Set water level boundary points

    # Loops around the water level boundary selecting points every 25,000m
    #sf.setup_waterlevel_bnd_from_mask(distance=25000,merge=False)
    sf.setup_waterlevel_bnd_from_mask(distance=20000,merge=False)

    _ = sf.plot_basemap(fn_out=f"{args.nwm_domain}_wl_bnd_pts.png", plot_region=False,bmap="sat",zoomlevel=12)

    # Step 11: Add observation points

    # Query lat lon points of NOAA stations within the open water boundary geojson
    REGION_GEOJSON = args.polygon_file
    BASE_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"

    def fetch_waterlevel_stations():
        """Fetch NOAA CO-OPS stations with water-level data."""
        r = requests.get(BASE_URL, params={"type": "waterlevels"}, timeout=30)
        r.raise_for_status()
        data = r.json().get("stations", [])
        return pd.DataFrame(
            [
                {
                "station_id": s["id"],
                "raw_name": s.get("name", ""),
                "lat": float(s["lat"]),
                "lon": float(s["lng"]),
                }
            for s in data
            ]
        )

    def get_noaa_waterlevel_stations(region_file):
        # 1. Load region polygon
        region = gpd.read_file(region_file).to_crs(4326)
        #region = gpd.read_file(region_file).to_crs(lcc)
        geom = region.geometry.unary_union

        # 2. Fetch water level stations
        df = fetch_waterlevel_stations()

        # 3. Format name as "Name (ID)"
        df["name"] = df["raw_name"] + " (" + df["station_id"] + ")"

        # 4. Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])],
            crs="EPSG:4326",
            #crs=lcc,
        )

        # 5. Spatial filter
        return gdf[gdf.within(geom)].copy()[["station_id", "name", "lon", "lat", "geometry"]]

    # ----------------------------
    # Usage
    # ----------------------------
    gdf_obs = get_noaa_waterlevel_stations(REGION_GEOJSON)

    sf.setup_observation_points(locations=gdf_obs)

    # Check that the observation points are stored in the sf.geoms dictionary
    sf.geoms.keys()

    # Plot the model setup to check it

    _ = sf.plot_basemap(fn_out=f"{args.nwm_domain}_mode_setup.png", bmap="sat",zoomlevel=12)

    # And save it

    sf.write()

    #create the sfincs.src file 

    reaches = SCHISMnwmReaches(os.path.join(args.nwmreach_file))

    schm_grid = SCHISMNcGrid( args.hgrid_file )

    with open(f"{args.model_root_dir}/sfincs_nwm.src", "w") as file:
       for e in reaches.soelem_ids:
         x,y = schm_grid.getElemCenterCoordsInCrs(e, f"EPSG:{epsg_code}" )
         file.write(f"{x:.6f} {y:.6f} {reaches.soelem_ids[e]}\n")

if __name__ == "__main__":
   try:
      main()
   except Exception as e:
      logging.error("Failed to get program options.", exc_info=True)
