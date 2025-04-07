"""
This module provides functions to create custom planes, optimized pipe paths,
and hollow pipe features using the Fusion 360 API.
All units are in centimeters.
"""

import traceback
import adsk.core
import adsk.fusion
import json
import os
import sys


# get the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# get the path of the the upper of the upper directory
# (the project root directory)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# add the project root directory to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)



def create_custom_plane(rootComp, point1, point2, point3=(-0.05,-0.04,-0.03)): # default point3; so that the function can be called with only 2 points
    """
    Creates a construction plane using three 3D points.

    Parameters:
    rootComp (adsk.fusion.Component): The root component of the active design.
    point1, point2, point3 (adsk.core.Point3D): Three points to define the plane.

    Returns:
    adsk.fusion.ConstructionPlane: The created custom construction plane.
    """
    sketches = rootComp.sketches
    xyPlane = rootComp.xYConstructionPlane
    sketch = sketches.add(xyPlane)

    sketchPoint1 = sketch.sketchPoints.add(point1)
    sketchPoint2 = sketch.sketchPoints.add(point2)
    sketchPoint3 = sketch.sketchPoints.add(point3)

    planes = rootComp.constructionPlanes
    planeInput = planes.createInput()
    planeInput.setByThreePoints(sketchPoint1, sketchPoint2, sketchPoint3)
    customPlane = planes.add(planeInput)

    return customPlane


def global_to_local(global_point, sketch):
    """
    Converts a global 3D point to the local coordinate system of a sketch.

    Parameters:
    global_point (adsk.core.Point3D): The point in global coordinates.
    sketch (adsk.fusion.Sketch): The sketch whose coordinate system to use.

    Returns:
    adsk.core.Point3D: The transformed point in sketch local coordinates.
    """
    parent_component = sketch.parentComponent
    normal = sketch.referencePlane.geometry.normal
    origin = sketch.referencePlane.geometry.origin

    x_direction = sketch.xDirection
    y_direction = normal.crossProduct(x_direction)

    transform = adsk.core.Matrix3D.create()
    transform.setWithCoordinateSystem(origin, x_direction, y_direction, normal)

    inverse_transform = transform.copy()
    inverse_transform.invert()

    local_point = global_point.copy()
    local_point.transformBy(inverse_transform)

    return local_point


def get_optimized_pipe(start, end):
    """
    Generates an optimized pipe path with one midpoint for smoothing.

    Parameters:
    start (adsk.core.Point3D): Start point of the pipe.
    end (adsk.core.Point3D): End point of the pipe.

    Returns:
    adsk.core.ObjectCollection: The collection of points defining the pipe path.
    """
    points = adsk.core.ObjectCollection.create()
    points.add(start)
    middle = adsk.core.Point3D.create(
        (start.x + end.x) / 2, (start.y + end.y) / 2, (start.z + end.z) / 2
    )
    points.add(middle)
    points.add(end)
    return points


def create_pipe_path(rootComp, feats, start, end, sketch):
    """
    Creates a spline-based pipe path on a given sketch.

    Parameters:
    rootComp (adsk.fusion.Component): The root component.
    feats (adsk.fusion.Features): The Features collection.
    start (adsk.core.Point3D): Global start point.
    end (adsk.core.Point3D): Global end point.
    sketch (adsk.fusion.Sketch): The sketch used to define the pipe path.

    Returns:
    adsk.fusion.Path: The created pipe path.
    """
    start_local = global_to_local(start, sketch)
    end_local = global_to_local(end, sketch)
    points = get_optimized_pipe(start_local, end_local)
    spline = sketch.sketchCurves.sketchFittedSplines.add(points)
    path = feats.createPath(spline)
    return path


def create_pipe(feats, path,isHollow = True, outDiameter = 0.8,  wallThickness=0.05):
    """
    Creates a hollow pipe with the specified outer diameter and wall thickness.

    Parameters:
    feats (adsk.fusion.Features): The Features collection of the current component.
    path (adsk.fusion.Path): The path along which the pipe is created.
    outDiameter (float): The outer diameter of the pipe in centimeters.
    wallThickness (float): The thickness of the pipe wall in centimeters.

    Returns:
    adsk.fusion.PipeFeature: The created pipe feature object.
    """
    pipeFeatures = feats.pipeFeatures
    pipeInput = pipeFeatures.createInput(
        path, adsk.fusion.FeatureOperations.NewBodyFeatureOperation
    )
    pipeInput.sectionSize = adsk.core.ValueInput.createByReal(outDiameter)
    if isHollow:
        pipeInput.isHollow = isHollow
        pipeInput.wallThickness = adsk.core.ValueInput.createByReal(wallThickness)
    pipeFeature = pipeFeatures.add(pipeInput)
    return pipeFeature



def read_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def create_spline_path(rootComp, points_3d):
    """
    Create a 3D sketch fitted spline from list of [x, y, z]
    """
    sketches = rootComp.sketches
    sketch = sketches.add(rootComp.xYConstructionPlane)  # 只是附着，不限制其在 XY 平面上

    point_collection = adsk.core.ObjectCollection.create()
    for pt in points_3d:
        point_collection.add(adsk.core.Point3D.create(*pt))

    spline = sketch.sketchCurves.sketchFittedSplines.add(point_collection)
    path = rootComp.features.createPath(spline)
    return path


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = app.activeProduct
        rootComp = design.rootComponent
        feats = rootComp.features
        
        print("Starting to create pipes process...")

        json_path = os.path.join(project_root, "exported_splines.json") # Path to the JSON file
        points_path = os.path.join(project_root, "paired_points.json") # Path to the paired points JSON file

        if not os.path.exists(json_path):
            ui.messageBox(f"File not found: {json_path}")
            return
        
        if not os.path.exists(points_path):
            ui.messageBox(f"File not found: {points_path}")
            return

        all_spline_data = read_json(json_path)
        pipe_radius = read_json(points_path)["pipe_radius"] # get the pipe radius from paired_points.json
        outDiameter = pipe_radius * 2 # outer diameter is twice the radius


        for idx, spline_pts in enumerate(all_spline_data):
            path = create_spline_path(rootComp, spline_pts)
            create_pipe(feats, path, isHollow=False, outDiameter=outDiameter)

        ui.messageBox("All splines converted to pipes successfully!")

    except:
        if ui:
            ui.messageBox("Failed:\n{}".format(traceback.format_exc()))