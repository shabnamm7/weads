#!/usr/bin/python
# coding: utf-8
# Provide general functions for point-based WEADS
# Developed by the Center for Computation & Technology and Center for Coastal Ecosystem Design Studio at Louisiana State University (LSU).
# Developer: Jin Ikeda, Peter Bacopoulos and Christopher E. Kees
# Last modified Nov 1, 2024

from netCDF4 import Dataset
from .KDTree_idw import Invdisttree
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import sys
import zipfile
import shutil
from scipy.spatial import KDTree
from datetime import datetime
from tqdm.auto import tqdm
import time
import glob
from pathlib import Path

# import spatial analysis
from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *
import rasterio
from rasterio.mask import mask
gdal.UseExceptions()


def read_text_file(fileName):
    with open(fileName, "r") as f:
        lines = f.readlines()
    return lines


def read_fort14(inputMeshFile, output_Flag=False):
    print("Processing mesh...")
    with open(inputMeshFile, "r") as f:
        lines = f.readlines()
    f.close()

    ##### Step.2 Get the number of nodes and elements  ######
    skip_index = 1  # skip the first line
    # eN:number of elemetns, #nN: number of nodes
    eN, nN = lines[skip_index].split()
    eN = int(eN)  # string to integer
    nN = int(nN)  # string to integer
    print("number of elements:eN={0},number of nodes:nN={1}".format(eN, nN))

    ##### Step.3 Output nodes ######
    nodes = []
    for i in range(nN):
        # skip eN and nN information
        nodeNum, x, y, z = lines[(skip_index + 1) + i].split()
        nodes.append([int(nodeNum), float(x), float(y),
                      float((-1) * float(z))])  # output longitude [degree],latitude [degree], and topobathy h [m, NAVD88] # Caution: ADCIRC z value (water depth direction is positive)

    ADCIRC_nodes = np.array(nodes)
    if output_Flag:
        np.savetxt(
            "mesh_original.txt",
            ADCIRC_nodes,
            fmt='%d\t%.8f\t%.8f\t%.8f')
    return ADCIRC_nodes, nN, eN


def read_fort13(inputMeshFile):
    with open(inputMeshFile, "r") as f:
        lines = f.readlines()
    f.close()

    skip_index = 1  # skip the first line
    nN = int(lines[skip_index].split()[0])  # nN: number of nodes

    # skip_index2 = 2  # read for the number of attributes
    # numAttributes = int(lines[skip_index2].split()[0])  # number of
    # attributes

    attribute = "mannings_n_at_sea_floor"
    mann_indices = []
    local_mann_indices = []
    global_mann = None  # initialize global_mann

    for i, line in enumerate(lines):
        if attribute in line:
            mann_indices.append(i)
            if not global_mann:  # if global_mann is not yet set
                global_mann = float(lines[i + 3].split()[0])
                print("global_manning\t", global_mann)
                # initialize using a global value
                mann = np.full((nN, 1), global_mann, dtype=float)
            else:
                local_mann_Num = int(lines[i + 1].split()[0])
                print("local_manning_Num\t", local_mann_Num)
                for j in range(local_mann_Num):
                    NN, value = map(float, lines[i + 2 + j].split())
                    local_mann_indices.append((int(NN) - 1))
                    mann[int(NN) - 1][0] = value  # overwrite the value

    return mann, mann_indices, local_mann_indices, global_mann


# Jin's comments. We also need to change the function to read netcdf file.
def read_inundationtime63(inputInundationTFile):
    # Convert inputInundationTFile to string if it's a PosixPath
    if isinstance(inputInundationTFile, Path):
        inputInundationTFile = str(inputInundationTFile)

    # Check if the input file is a NetCDF file
    if ".nc" in inputInundationTFile:
        # Open the NetCDF file
        ds = Dataset(inputInundationTFile, mode='r')

        # Check the contents of the file
        # print(ds)  # if the user want to see the contents, turn on this
        # command

        # Access specific variables
        time = ds.variables['time'][:]
        x = ds.variables['x'][:]
        y = ds.variables['y'][:]
        inun_time = ds.variables['inun_time'][:]

        nN = len(x)  # nN: number of nodes
        max_time = float(time[0])  # maximum simulation time

        print("node number\t", nN, "max_time\t", max_time)

        # Initialize the array for inundation time
        inundationtime = np.zeros(
            nN, dtype=[('nodeNum', int), ('time', float)])

        # Assign node number to 'nodeNum' field
        inundationtime['nodeNum'] = np.arange(1, nN + 1)

        # Assign inundationtime to 'time' field
        inundationtime['time'] = inun_time / max_time

        # Close the NetCDF file
        ds.close()

    else:
        # Open and read the text file
        with open(inputInundationTFile, "r") as f:
            lines = f.readlines()
        f.close()

        ##### Step.2 Get the number of nodes and maximum simulation time ######
        skip_index = 1  # skip the first line
        nN = int(lines[skip_index].split()[1])  # nN: number of nodes
        skip_index2 = 2  # skip the line
        # max_time: maximum simulation time [s]
        max_time = float(lines[skip_index2].split()[1])
        print("node number\t", nN, "max_time\t", max_time)

        ##### Step.3 Output inundationtime ######
        # node number, time of inundation (0: dry, 1: wet)
        inundationtime = np.zeros(
            nN, dtype=[('nodeNum', int), ('time', float)])
        inundationtime = np.zeros(
            nN, dtype=[('nodeNum', int), ('time', float)])
        for i in range(nN):
            nodeNum, time = lines[(skip_index2 + 1) + i].split()
            inundationtime[i]['nodeNum'] = int(nodeNum)
            if float(time) >= 0:
                # float(float(time) / max_time)  # time of inundation (0: dry, 1:
                # wet)
                inundationtime[i]['time'] = float(time) / max_time
            else:
                inundationtime[i]['time'] = 0

    return inundationtime, nN, max_time


def read_max_inundationdepth63(inputMaxdepthFile):
    """
    Reads the max inundation depth from a NetCDF file and returns an array with node number, longitude, latitude, and depth.

    :param inputMaxdepthFile: Path to the NetCDF file.
    :return: A structured numpy array with fields 'nodeNum', 'x', 'y', and 'depth'.
    """
    if isinstance(inputMaxdepthFile, Path):
        inputInundationTFile = str(inputMaxdepthFile)

    # Check if the input file is a NetCDF file
    if ".nc" in inputMaxdepthFile:
        # Open the NetCDF file
        ds = Dataset(inputMaxdepthFile, mode='r')
        # Check the contents of the file
        #print(ds)  # if the user want to see the contents, turn on this command
        x = ds.variables['x'][:]
        y = ds.variables['y'][:]
        depth = ds.variables['inun_max'][:]
        nN = len(x)
        inundationdepth = np.zeros(
            nN, dtype=[('nodeNum', int), ('x', float), ('y', float), ('depth', float)]
        )

        inundationdepth['nodeNum'] = np.arange(1, nN + 1)
        inundationdepth['x'] = x
        inundationdepth['y'] = y
        inundationdepth['depth'] = depth

        ds.close()

    else:
        with open(inputMaxdepthFile, "r") as f:
            lines = f.readlines()
        f.close()

        ##### Step.2 Get the number of nodes and maximum inundation depth ######
        skip_index = 1  # skip the first line
        nN = int(lines[skip_index].split()[1])  # nN: number of nodes
        skip_index2 = 2  # skip the line

        print("Need to confirm the format of the text file (Jin 10/21/2024)")

    return inundationdepth

def read_maxele(inputMaxeleFile):
    # Convert inputInundationTFile to string if it's a PosixPath
    if isinstance(inputMaxeleFile, Path):
        inputMaxeleFile = str(inputMaxeleFile)

    # Check if the input file is a NetCDF file
    if ".nc" in inputMaxeleFile:
        # Open the NetCDF file
        ds = Dataset(inputMaxeleFile, mode='r')

        # Check the contents of the file
        print(ds)  # if the user want to see the contents, turn on this command

        # Access specific variables
        x = ds.variables['x'][:]
        y = ds.variables['y'][:]
        zeta_max = ds.variables['zeta_max'][:]
        depth = ds.variables['depth'][:]

        nN = len(x)  # nN: number of nodes
        print("node number\t", nN)

        # Initialize the array for inundation depth
        inundationdepth = np.zeros(
            nN, dtype=[('nodeNum', int), ('x', float), ('y', float),('zeta_max',float),('depth', float)]
        )

        inundationdepth['nodeNum'] = np.arange(1, nN + 1)
        inundationdepth['x'] = x
        inundationdepth['y'] = y
        inundationdepth['zeta_max'] = zeta_max
        inundationdepth['depth'] = depth

        ds.close()

    else:
        # Need to confirm the format of the text file (Jin 10/21/2024)
        # Open and read the text file
        with open(inputMaxeleFile, "r") as f:
            lines = f.readlines()
        f.close()

        ##### Step.2 Get the number of nodes and maximum simulation time ######
        skip_index = 1  # skip the first line
        nN = int(lines[skip_index].split()[1])  # nN: number of nodes
        skip_index2 = 2  # skip the line
        # max_ele: maximum simulation time [s]
        max_ele = float(lines[skip_index2].split()[1])
        print("node number\t", nN, "max_time\t", max_ele)

        # ##### Step.3 Output inundationtime ######
        # # node number, time of inundation (0: dry, 1: wet)
        # inundationdepth = np.zeros(
        #     nN, dtype=[('nodeNum', int), ('time', float)])

        print("Need to confirm the format of the text file (Jin 10/21/2024)")

    return inundationdepth

def read_fort53(inputHarmonicsFile):
    print("   Processing harmonics...\n")
    # Jin's note: We may need to change the function to read netcdf file.
    with open(inputHarmonicsFile, "r") as f:
        lines = f.readlines()
    f.close()

    ##### Step.2 Get the number of nodes and maximum simulation time ######
    skip_index = 0  # skip the first line
    numHarm = int(lines[skip_index].split()[0])  # numHarm: number of Harmonics

    tidal_frequncies = []
    tidal_constitunents = []

    skip_index2 = 1  # skip the first line
    for i in range(numHarm):
        # tidal frequencies (rad/s)
        tidal_frequncies.append(float(lines[skip_index2 + i].split()[0]))
        tidal_constitunents.append(
            lines[skip_index2 + i].split()[3])  # tidal constituents

    skip_index3 = 1 + numHarm  # skip the line until the number of nodes
    nN = int(lines[skip_index3].split()[0])  # nN: number of nodes
    print("number of Harmonics\t", numHarm, ", node number\t", nN)

    # Tidal harmonics of each node
    # 3D array [AMP, PHASE] cannot save as a text file
    Harmonics_nodes = np.zeros((nN, numHarm, 2), dtype=float)

    for i in range(nN):
        # print('check',lines[skip_index3 + 1 + i * (numHarm + 1)].split())
        for ii in range(numHarm):
            # skip before EMAGT(k,j), PHASEDE(k,j) here number of node is
            # skipped.
            AMP, PHASE = lines[skip_index3 + 2 +
                               i * (numHarm + 1) + ii].split()
            Harmonics_nodes[i][ii][0] = float(AMP)
            Harmonics_nodes[i][ii][1] = float(PHASE)
            # print(ii, Harmonics_nodes[i][ii][0], Harmonics_nodes[i][ii][1])

    return Harmonics_nodes, nN, numHarm, tidal_frequncies, tidal_constitunents


def read_domain(inputShapeFile):
    gdf = gpd.read_file(inputShapeFile)
    return gdf


def create_gdf(ADCIRC_nodes, xy_list, drop_list, crs, output_file=None):
    # Convert the numpy array to a pandas DataFrame
    df = pd.DataFrame(ADCIRC_nodes, columns=["node_id", "x", "y", "z"])
    # Or Read the text file into a pandas DataFrame
    # df = pd.read_csv(ADCIRC_Mesh_File, sep="\t", header=None, names=["node_id", "x", "y", "z"])
    gdf = gpd.GeoDataFrame(df.drop(drop_list, axis=1), geometry=gpd.points_from_xy(
        df[xy_list[0]], df[xy_list[1]], crs=crs))

    if output_file is not None:
        gdf.to_file(output_file, driver='ESRI Shapefile')

    return gdf


def create_df2gdf(df, xy_list, drop_list, crs, output_file=None):
    gdf = gpd.GeoDataFrame(df.drop(drop_list, axis=1), geometry=gpd.points_from_xy(
        df[xy_list[0]], df[xy_list[1]], crs=crs))

    if output_file is not None:
        gdf.to_file(output_file, driver='ESRI Shapefile')

    return gdf


# Type_str means convert the type of vector data.
def convert_gcs2coordinates(gdf, PRJ, Type_str):
    gdf_proj = gdf.to_crs(PRJ)
    if Type_str == "Point":
        gdf_proj["x_prj"] = gdf_proj.geometry.apply(lambda point: point.x)
        gdf_proj["y_prj"] = gdf_proj.geometry.apply(lambda point: point.y)
    elif Type_str == "Polygon":
        gdf_proj["x"] = gdf_proj.geometry.apply(
            lambda polygon: polygon.centroid.x)
        gdf_proj["y"] = gdf_proj.geometry.apply(
            lambda polygon: polygon.centroid.y)
    else:
        print("Type_str is not defined")
    return gdf_proj


def convert_coordinates2gcs(gdf_proj, PRJ):
    gdf = gdf_proj.to_crs(PRJ)
    gdf["x"] = gdf.geometry.apply(lambda point: point.x)
    gdf["y"] = gdf.geometry.apply(lambda point: point.y)
    return gdf


def filter_points_within_domain(points_gdf, polygon_gdf):
    # Perform a spatial join between the points and the polygons
    points_within_domain = points_gdf[points_gdf.geometry.within(
        polygon_gdf.unary_union)]

    # Get the indices of the rows in points_within_domain
    true_indices = points_within_domain.index

    return points_within_domain, true_indices


def gdal_reading(file):  # 0 GA_ReadOnly or 1
    Rasterdata = gdal.Open(file, GA_ReadOnly)
    if Rasterdata is None:
        print("Could not open " + Rasterdata)
        sys.exit(1)
    print("Reading raster file (georeference, etc)")

    # Coordinate system
    prj = Rasterdata.GetProjection()  # Read projection
    print("Projection:", prj)

    # Get raster size and band
    rows = Rasterdata.RasterYSize  # number of rows
    cols = Rasterdata.RasterXSize  # number of columns
    bandnum = Rasterdata.RasterCount  # band number
    print("rows=", rows, "cols=", cols)
    # print("band=", bandnum)

    # Get georeference info
    transform = Rasterdata.GetGeoTransform()
    xOrigin = transform[0]  # Upperleft x
    yOrigin = transform[3]  # Upperleft y
    pixelWidth = transform[1]  # cell size x
    pixelHeight = transform[5]  # cell size y (value is negative)
    print("xOrigin=", xOrigin, "m", "yOrigin=", yOrigin, "m")
    print("pixelWidth=", pixelWidth, "m", "pixelHeight=", -
          pixelHeight, "m")  # pixelHeight is always negative

    # Read the raster band
    band = Rasterdata.GetRasterBand(1)
    # Data type of the values
    # Each raster file has a different data type
    print('data type is', gdal.GetDataTypeName(band.DataType))
    # Get band value info
    RV = band.ReadAsArray()          # raster values in the band

    return prj, rows, cols, transform, RV, Rasterdata


def dummy_raster(xx, yy, epsg_code, nodata_value=None):
    """
    Creates a temporary GeoTIFF raster file for testing, using grid coordinates.

    :param xx: 2D array of X coordinates.
    :param yy: 2D array of Y coordinates.
    :param epsg_code: EPSG code for the raster's spatial reference system.
    :param nodata_value: Value to use for no-data pixels.
    :return: File path to the created raster.
    """
    file_path = "dummy_raster.tif"

    # Determine the number of rows and columns from the grid dimensions
    rows, cols = xx.shape

    # Calculate the top left coordinates and pixel size
    top_left_x = xx[0, 0]
    top_left_y = yy[-1, 0]  # last row should be top
    pixel_size_x = abs(xx[0, 1] - xx[0, 0])
    pixel_size_y = abs(yy[0, 0] - yy[1, 0])

    # Initialize data array with nodata_value
    data_array = np.full((rows, cols), 10 ** -6, dtype=np.float32)

    # Set up the GeoTransform
    transform = (top_left_x, pixel_size_x, 0, top_left_y, 0, -pixel_size_y)

    # Set up the projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    projection = srs.ExportToWkt()

    # Create a GeoTIFF file
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(str(file_path), cols, rows, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(transform)
    out_ds.SetProjection(projection)

    # Write the data
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(data_array)
    if nodata_value is not None:
        out_band.SetNoDataValue(nodata_value)

    # Close the dataset to flush data to disk
    out_ds = None

    return file_path


def extract_point_values(raster_path, points_gdf, points_path, ndv, ndv_byte):
    # Load the shapefile of points
    points = points_gdf
    # Load the DEM raster
    dem = rasterio.open(raster_path)

    # extract xy from point geometry
    raster_id = np.zeros(len(points))
    raster_values = np.full(len(points), ndv_byte)

    array = dem.read(1)
    for i, point in enumerate(points.geometry):
        # print(point.xy[0][0],point.xy[1][0])
        x = point.xy[0][0]
        y = point.xy[1][0]
        row, col = dem.index(x, y)

        # Append the z value to the list of z values
        if 0 <= row < array.shape[0] and 0 <= col < array.shape[1]:
            raster_id[i] = row * array.shape[1] + col
            raster_values[i] = array[row, col]
        else:
            raster_id[i] = ndv

        # print("Point correspond to row, col: %d, %d"%(row,col))
        # print(array[row, col])
        # print("Raster value on point %.2f \n"%dem.read(1)[row,col])

    points['Raster_id'] = raster_id
    points['NWI'] = raster_values
    # points.to_file(points_path, driver='ESRI Shapefile')
    # Save the DataFrame to a CSV file with headers
    points.to_csv(points_path, index=False)

    return raster_values, points


def expand_nodes(nodes_positions, node_states,
                 target_value, infection_distance):
    # nodes_positions: the positions of the nodes (x, y coordinates) as a numpy array (N,2)
    # node_states: the states of the nodes (0: not infected, target_value:
    # infected) as a numpy array (N,1)

    # Create a KDTree for efficient neighbor search
    tree = KDTree(nodes_positions)

    # Find all nodes within the infection distance of the infected node(s) and
    # update their states
    # Get the indices of infected nodes here node_state a tuple
    potential_expansion_nodes = np.where(node_states == target_value)[0]
    for node in potential_expansion_nodes:
        neighbors = tree.query_ball_point(
            nodes_positions[node],
            infection_distance,
            p=2.0)  # p=2 (circle) for Euclidean distance
        for neighbor in neighbors:
            if neighbor != node:  # Exclude the node itself
                node_states[neighbor] = target_value

    return node_states


# internal function of generate_grid
def read_shp_extent(inputShapeFile, gcs2prj=False, inEPSG=None, outEPSG=None):
    """
    Reads the extent of a shapefile using geopandas and optionally converts the coordinate system.

    :param inputShapeFile: Path to the shapefile.
    :param gcs2prj: Flag to convert from GCS to projected coordinate system.
    :param inEPSG: EPSG code of the input coordinate system (if known).
    :param outEPSG: EPSG code of the output projected coordinate system (if gcs2prj is True).
    :return: Tuple containing the extent and bounding box.
    """

    # Read the shapefile
    gdf = read_domain(inputShapeFile)

    # Reproject if necessary
    if gcs2prj and inEPSG and outEPSG:
        gdf = gdf.to_crs(epsg=outEPSG)

    # Get the bounding box (extent)
    extent = gdf.total_bounds
    print("\tExtent of points id: %s", extent)

    # Create bbox corners based on extent
    bbox = [(extent[0], extent[1]), (extent[2], extent[1]),
            (extent[2], extent[3]), (extent[0], extent[3])]

    # extent[0] = minx
    # extent[1] = miny
    # extent[2] = maxx
    # extent[3] = maxy
    # bbox = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]

    return extent, bbox


def generate_grid(inputShapeFile, resolution=20, scale_factor=1, gcs2prj=False,
                  inEPSG=None, outEPSG=None):
    """
    Generate a grid of X and Y coordinates based on the shapefile extent and desired resolution.

    :param inputShapeFile: Path to the shapefile to read the extent.
    :param resolution: Grid resolution (default is 20 meters).
    :param scale_factor: Scale factor for the grid (default is 1).
    :param gcs2prj: Boolean indicating if the coordinate system should be projected (default is False).
    :param inEPSG: EPSG code of the input coordinate system (default is None).
    :param outEPSG: EPSG code of the output projected coordinate system (default is None).
    :return: X and Y coordinate grids.
    """

    # Read shapefile extent
    extent, bbox = read_shp_extent(inputShapeFile, gcs2prj, inEPSG, outEPSG)
    print(extent, bbox)

    scale_factor = int(scale_factor)
    gridx = np.arange(int(extent[0] *
                          scale_factor -
                          resolution /
                          2), int(extent[2] *
                                  scale_factor +
                                  resolution /
                                  2), resolution)  # draw region will be expanded
    gridy = np.arange(int(extent[1] *
                          scale_factor -
                          resolution /
                          2), int(extent[3] *
                                  scale_factor +
                                  resolution /
                                  2), resolution)

    gridx = gridx / scale_factor
    gridy = gridy / scale_factor

    xx, yy = np.meshgrid(gridx, gridy)  # grid_x and y

    return xx, yy


# innternal function for interpolated grid
def write_clipped_raster(raster_open, raster, clip_file,
                         transform, nodata_value=-99999.0):
    out_meta = raster_open.meta.copy()
    out_meta.update({
        "height": raster.shape[1],
        "width": raster.shape[2],
        "transform": transform,
        "nodata": nodata_value
    })

    # clip_file = f"{Outputspace}/{method_str}_train_{i}_{date}.tif"
    with rasterio.open(clip_file, "w", **out_meta) as dest:
        dest.write(raster)

    return clip_file

def should_use_idw_from_dtype(dtype_value):
    """
    Use IDW for floating-point rasters and nearest-neighbor-like interpolation
    for integer / thematic rasters.
    """
    float_types = {
        gdal.GDT_Float32,
        gdal.GDT_Float64,
    }
    return dtype_value in float_types

def interpolate_grid(xx, yy, df, target_list, inputShapeFile, ref_tiff, output_file, idw_Flag=None, knn=12,
                     dtype_list=None, nodata_value_list=None, reproject_flag=False, inEPSG=None, outEPSG=None, mask_flag=True):
    """
    Interpolates values and creates a single GeoTIFF raster file with multiple bands for the target strings.
    :param xx: 2D numpy array of x coordinates.
    :param yy: 2D numpy array of y coordinates.
    :param df: DataFrame containing the input data with 'x', 'y', and target columns.
    :param target_list: List of target strings (column names in df) to interpolate.
    :param ref_tiff: Reference GeoTIFF file path for geotransform and projection.
    :param output_files: List of output file paths for the resulting rasters.
    :param idw_Flag: Boolean flag to use IDW interpolation or nearest neighbor.
    :param knn: Number of nearest neighbors to use in interpolation.
    :param dtype: Data type for the output raster.
    :param nodata_value: Value to use for no-data pixels.
    :param reproject_flag: Flag to reproject coordinates.
    :param inEPSG: Input EPSG code if reprojection is needed.
    :param outEPSG: Output EPSG code if reprojection is needed.
    :param mask_flag: Flag to mask the raster using the shapefile. Defaults to True.
    """

    try:
        # Check input parameters
        assert isinstance(xx, np.ndarray) and isinstance(
            yy, np.ndarray), "xx and yy must be numpy arrays"
        assert xx.shape == yy.shape, "xx and yy must have the same shape"
        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        assert isinstance(target_list, list), "target_list must be a list"
        assert 'x' in df.columns and 'y' in df.columns, "df must contain 'x', 'y' columns"
        assert isinstance(
            knn, int) and knn > 0, "knn must be a positive integer"

        domain_shp = gpd.read_file(inputShapeFile)

        # Interpolation using KDTree
        if reproject_flag:
            assert inEPSG is not None and outEPSG is not None, "Provide inEPSG and outEPSG"
            crs = 'EPSG:' + str(inEPSG)
            PRJ = int(outEPSG)
            gdf = create_df2gdf(df, ['x', 'y'], [], crs, None)
            gdf_proj = convert_gcs2coordinates(gdf, PRJ, "Point")
            mask_shp = convert_gcs2coordinates(domain_shp, PRJ, "Polygon")
            xy_base = gdf_proj[['x_prj', 'y_prj']].to_numpy()
        else:
            xy_base = df[['x', 'y']].to_numpy()
            mask_shp = domain_shp.copy()

        # Initialize a list to hold the interpolated grids for each target
        interpolated_grids = []

        if dtype_list is None:
            raise ValueError("dtype_list must be provided")

        dtype0 = dtype_list[0]

        # If not explicitly specified, choose interpolation automatically
        # from output raster dtype.
        if idw_Flag is None:
            idw_Flag = should_use_idw_from_dtype(dtype0)
            print(f"Auto-selected interpolation for {output_file}: "
                  f"{'IDW' if idw_Flag else 'NN'} based on dtype {dtype0}")

        for target_str in target_list:
            assert target_str in df.columns, f"df must contain {target_str} column"

            z_base = df[target_str].values
            invdisttree = Invdisttree(xy_base, z_base, leafsize=10, stat=1)
            interp_grids = np.transpose(np.vstack([xx.ravel(), yy.ravel()]))

            if idw_Flag:
                interpolated_values, _ = invdisttree(
                    interp_grids, nnear=knn, eps=0.0, p=2)
            else:
                interpolated_values, _ = invdisttree(
                    interp_grids, nnear=1, eps=0.0, p=2)

            z_interp = np.reshape(
                interpolated_values, (xx.shape[0], xx.shape[1]))
            interpolated_grids.append(z_interp)

        # Stack all interpolated grids along the third dimension
        raster_stack = np.dstack(interpolated_grids)

        # Create a raster with all interpolated values as multiple bands
        rasterdata = gdal.Open(ref_tiff, 0)
        temp_file = "process.tif"
        create_raster(
            temp_file,
            rasterdata,
            raster_stack,
            dtype_list,
            nodata_value_list)

        # Open the TIFF file and clip using a polygon
        print("clip raster file")

        nodata_value = nodata_value_list[0]  # only use first value
        if mask_flag:
            with rasterio.open(temp_file) as raster_open:
                raster = raster_open.read(1)
                transform = raster_open.transform

                try:
                    if np.issubdtype(raster_open.dtypes[0], np.integer):
                        raster, transform = mask(raster_open, mask_shp.geometry, crop=True, all_touched=True,
                                                 nodata=nodata_value)
                    else:
                        raster, transform = mask(raster_open, mask_shp.geometry, crop=True, all_touched=True,
                                                 nodata=np.nan)

                except Exception as e:
                    print(f"Error in mask: {e}")
                    return None

                    # Ensure nodata_value is consistent with the raster dtype
                    # when writing
                if np.issubdtype(raster_open.dtypes[0], np.integer):
                    write_clipped_raster(
                        raster_open,
                        raster,
                        output_file,
                        transform,
                        nodata_value=int(nodata_value))
                else:
                    write_clipped_raster(
                        raster_open,
                        raster,
                        output_file,
                        transform,
                        nodata_value=float(nodata_value))

    except Exception as e:
        print(f"Error in interpolate_grid: {e}")
        return None


def create_raster(file, rasterdata, zarray_stack,
                  dtype_list, nodata_value_list):
    """
    Create a multi-band raster from stacked 2D arrays.
    """
    print(file)
    gtiff_driver = gdal.GetDriverByName('GTiff')
    num_bands = zarray_stack.shape[2]
    print(f"number of band", num_bands)
    out_ds = gtiff_driver.Create(
        file,
        rasterdata.RasterXSize,
        rasterdata.RasterYSize,
        num_bands,
        dtype_list[0]  # Assuming all bands have the same dtype
    )
    out_ds.SetProjection(rasterdata.GetProjection())
    out_ds.SetGeoTransform(rasterdata.GetGeoTransform())

    print("start working")
    for i in range(num_bands):
        print(i)
        band_data = zarray_stack[:, :, i]
        dtype = dtype_list[0]
        nodata_value = nodata_value_list[0]

        out_band = out_ds.GetRasterBand(i + 1)
        # bottom row <-> top row due to origin of raster file
        out_band.WriteArray(np.flipud(band_data))
        out_band.SetNoDataValue(nodata_value)

        stats = out_band.ComputeStatistics(0)
        print(
            f'Band {i+1} Statistics: Min: {stats[0]}, Max: {stats[1]}, Mean: {stats[2]}, StdDev: {stats[3]}')

    out_ds = None


def delete_files(file_list):

    # Delete file
    print("\nDelete files\n")

    for pattern in file_list:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"{file} has been deleted.")
            except FileNotFoundError:
                print(f"{file} not found.")
            except PermissionError:
                print(f"{file} cannot be deleted due to permission error.")


def update_ADCIRC_mesh(outputMeshFile, node_id, x, y, new_z):
    """
    Updates the z-values in an ADCIRC mesh file based on input arrays.

    Parameters:
    - outputMeshFile (str): Path to the ADCIRC mesh file to be updated.
    - node_id (array-like): Array of node indices.
    - x (array-like): X-coordinates of the nodes.
    - y (array-like): Y-coordinates of the nodes.
    - new_z (array-like): New z-values to update in the mesh file.

    """
    # Read the entire file into a list of lines
    lines = read_text_file(outputMeshFile)

    # Update the z value for the ADCIRC mesh
    for i in range(len(node_id)):
        idx = int(node_id[i])
        # Update the line corresponding to the node index
        # Node indices in the file are assumed to start from line 2 (index 1 in Python)
        lines[idx + 1] = "{0:>10}     {1:.6f}      {2:.6f} {3:.8E}\n".format(
            idx, x[i], y[i], -new_z[i])

    # Determine the output file path
    output_path = outputMeshFile

    # Write the updated lines back to the file
    with open(output_path, 'w') as file:
        file.writelines(lines)
    file.close()

    print(f"Updated ADCIRC file '{output_path}' with new node data from mem results")
