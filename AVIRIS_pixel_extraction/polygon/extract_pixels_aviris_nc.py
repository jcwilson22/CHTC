#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_pixels_aviris_nc.py

Per-pixel hyperspectral extraction from AVIRIS-NG netCDF imagery
using polygon geometries.
"""

import os
import sys
import argparse
import csv
import gzip
from math import floor, ceil

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()

print("Running 3/19 extract_pixels_aviris_nc.py script")

# ---------------------------------------------------------------------
# Geotransform helpers
# ---------------------------------------------------------------------
def invert_geotransform(gt):
    originX, pxW, rotX, originY, rotY, pxH = gt
    det = pxW * pxH - rotX * rotY
    if abs(det) < 1e-12:
        raise RuntimeError("Non-invertible geotransform")

    a =  pxH / det
    b = -rotX / det
    d = -rotY / det
    e =  pxW / det
    c = -(a * originX + b * originY)
    f = -(d * originX + e * originY)

    return [[a, b, c],
            [d, e, f]]

def geom_to_pixel_window(geom, gt, cols, rows):
    xmin, xmax, ymin, ymax = geom.GetEnvelope()

    px_min = int((xmin - gt[0]) / gt[1])
    px_max = int((xmax - gt[0]) / gt[1])

    py_min = int((ymax - gt[3]) / gt[5])
    py_max = int((ymin - gt[3]) / gt[5])

    xoff = max(0, min(px_min, px_max))
    yoff = max(0, min(py_min, py_max))

    xend = min(cols, max(px_min, px_max))
    yend = min(rows, max(py_min, py_max))

    xsize = xend - xoff
    ysize = yend - yoff

    if xsize <= 0 or ysize <= 0:
        return None

    return (xoff, yoff, xsize, ysize)

def rasterize_polygon_mask(geom, xsize, ysize, gt_win, proj_wkt):
    mem_drv = gdal.GetDriverByName("MEM")
    mask_ds = mem_drv.Create("", xsize, ysize, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(gt_win)
    mask_ds.SetProjection(proj_wkt)

    mem_ogr = ogr.GetDriverByName("Memory")
    ogr_ds = mem_ogr.CreateDataSource("mem")
    layer = ogr_ds.CreateLayer("poly", srs=osr.SpatialReference(wkt=proj_wkt))

    feat_def = layer.GetLayerDefn()
    feat = ogr.Feature(feat_def)
    feat.SetGeometry(geom.Clone())
    layer.CreateFeature(feat)

    gdal.RasterizeLayer(
    mask_ds,
    [1],
    layer,
    burn_values=[1],
    options=["ALL_TOUCHED=TRUE"]
    )
    
    mask = mask_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

    feat = None
    layer = None
    ogr_ds = None
    mask_ds = None

    return mask

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(args):

    # ---- Open AVIRIS-NG reflectance ----
    image_id = os.path.basename(args.image).replace(".nc", "")
    flight_date = image_id[3:11]
    flight_time = image_id[12:18]

    subds = f'NETCDF:"{args.image}":/reflectance/reflectance'
    ds = gdal.Open(subds, gdal.GA_ReadOnly)

    if ds is None:
        raise RuntimeError(f"Could not open reflectance dataset: {subds}")

    cols = ds.RasterXSize
    rows = ds.RasterYSize
    bands = ds.RasterCount
    gt = ds.GetGeoTransform()
    proj_wkt = ds.GetProjection()
    gt_inv = invert_geotransform(gt)
    print("Image X range:", gt[0], gt[0] + cols * gt[1])
    print("Image Y range:", gt[3] + rows * gt[5], gt[3])

    # ---- Open shapefile ----
    shp_ds = ogr.Open(args.shapefile)
    if shp_ds is None:
        raise RuntimeError(f"Could not open shapefile: {args.shapefile}")

    layer = shp_ds.GetLayer(0)

    print("Shapefile CRS WKT:")
    print(layer.GetSpatialRef().ExportToPrettyWkt())

    layer_defn = layer.GetLayerDefn()

    print("Available fields:")
    for i in range(layer_defn.GetFieldCount()):
        print(layer_defn.GetFieldDefn(i).GetName())

    # -----------------------------------------------------------------
    # FIXED CRS HANDLING
    # -----------------------------------------------------------------
    # ---- FORCE CORRECT CRS HANDLING ----
    img_srs = osr.SpatialReference()
    img_srs.ImportFromWkt(proj_wkt)
    img_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    source_srs = osr.SpatialReference()
    source_srs.ImportFromEPSG(4326)
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    coord_tx = osr.CoordinateTransformation(source_srs, img_srs)

    field_names = [
        layer_defn.GetFieldDefn(i).GetName()
        for i in range(layer_defn.GetFieldCount())
    ]

    # ---- Output CSV ----
    csvfile = gzip.open(args.out_csv, "wt", newline="")
    writer = csv.writer(csvfile)

    header = (
        ["pixel_id", "polygon_id", "image_id", "flight_date", "flight_time"] +
        field_names +
        ["pixel_row", "pixel_col", "x", "y"] +
        [f"band_{i+1}" for i in range(bands)]
    )
    writer.writerow(header)

    # ---- Loop polygons ----
    layer.ResetReading()

    for feat in layer:
        fid = feat.GetFID()
        geom = feat.GetGeometryRef()

        if geom is None:
            continue

        geom = geom.Clone()

        # -------------------------------------------------------------
        # AUTO-DETECT + TRANSFORM IF LAT/LON
        # -------------------------------------------------------------
        xmin, xmax, ymin, ymax = geom.GetEnvelope()

        if abs(xmin) < 180 and abs(ymin) < 90:
            geom.Transform(coord_tx)

        # Debug
        xmin, xmax, ymin, ymax = geom.GetEnvelope()
        print("Transformed bounds:", xmin, xmax, ymin, ymax)

        print("Image size:", cols, rows)
        print("GeoTransform:", gt)

        field_lookup = {fn.lower(): fn for fn in field_names}
        requested = args.polygon_id_field.lower()

        if requested not in field_lookup:
            raise RuntimeError(
                f"Polygon ID field '{args.polygon_id_field}' not found.\n"
                f"Available fields: {field_names}"
            )

        polygon_id = feat.GetField(field_lookup[requested])

        if polygon_id is None:
            raise RuntimeError(
                f"Polygon FID={fid} missing ID field '{args.polygon_id_field}'"
            )

        attrs = {fn: feat.GetField(fn) for fn in field_names}

        win = geom_to_pixel_window(geom, gt, cols, rows)
        print("Window:", win)

        if win is None:
            continue

        xoff, yoff, xsize, ysize = win

        gt_win = (
            gt[0] + xoff * gt[1],
            gt[1],
            0.0,
            gt[3] + yoff * gt[5],
            0.0,
            gt[5],
        )

        data = np.empty((bands, ysize, xsize), dtype=np.float32)
        for b in range(bands):
            data[b] = ds.GetRasterBand(b + 1).ReadAsArray(
                xoff, yoff, xsize, ysize
            )

        mask = rasterize_polygon_mask(geom, xsize, ysize, gt_win, proj_wkt)

        print(f"Window: {xoff},{yoff},{xsize},{ysize} | Mask sum: {mask.sum()}")
        print(f"Mask sum: {mask.sum()}")
        print(f"Window size: {xsize} x {ysize}") 

        ys, xs = np.nonzero(mask)

        for i in range(len(ys)):
            ry, cx = int(ys[i]), int(xs[i])
            global_row = yoff + ry
            global_col = xoff + cx

            x = gt[0] + global_col * gt[1] + global_row * gt[2] + 0.5 * gt[1]
            y = gt[3] + global_col * gt[4] + global_row * gt[5] + 0.5 * abs(gt[5])

            pixel_id = f"{polygon_id}_{image_id}_{global_row}_{global_col}"

            row = [
                pixel_id,
                polygon_id,
                image_id,
                flight_date,
                flight_time,
            ]
            row += [attrs.get(fn) for fn in field_names]
            row += [global_row, global_col, float(x), float(y)]

            for b in range(bands):
                val = data[b, ry, cx]
                row.append("" if np.isnan(val) else float(val))

            writer.writerow(row)

    csvfile.close()
    ds = None
    shp_ds = None

# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--shapefile", required=True)
    parser.add_argument("--polygon_id_field", required=True)
    parser.add_argument("--out_csv", default="extracted_pixels.csv.gz")

    args = parser.parse_args()
    main(args)