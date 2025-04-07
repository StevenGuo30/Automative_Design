"""
This module provides functions to create custom planes, optimized pipe paths,
and hollow pipe features using the Fusion 360 API.
All units are in centimeters.
"""

import traceback
import adsk.core
import adsk.fusion
import yaml
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))  # script directory
sys.path.append("../../fusion_api_example")
yaml_path = os.path.join(script_dir, "paired_points.yaml")

from Generate_path import generate_pipe_paths


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


def create_pipe(feats, path, outDiameter, wallThickness):
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
    pipeInput.isHollow = True
    pipeInput.wallThickness = adsk.core.ValueInput.createByReal(wallThickness)
    pipeFeature = pipeFeatures.add(pipeInput)
    return pipeFeature

def read_points():
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        point_dict = {}
        group_connections = []
        group = []
        for group in data:
            for point in group:
                point_dict[point["name"]] = [point["coordinates"]["x"], point["coordinates"]["y"], point["coordinates"]["z"]]
                group.append(point["name"])
            group_connections.append(group)
        return point_dict, group_connections

def run(context):
    """
    Main entry point to create two pipes with defined endpoints and custom plane.
    """
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = app.activeProduct
        rootComp = design.rootComponent
        ui.messageBox("Creating pipes...")
        if not design:
            ui.messageBox("No active design")
            return

        feats = rootComp.features
        sketches = rootComp.sketches
        point_dict, group_connections = read_points()

        _,points,edges = generate_pipe_paths(point_dict, group_connections)

        for edge in edges:
            point1=adsk.core.Point3D.create(*points[edge[0]])
            point2=adsk.core.Point3D.create(*points[edge[1]])
            customPlane = create_custom_plane(rootComp, point1, point2)
            sketch = sketches.add(customPlane)
            path = create_pipe_path(rootComp, feats, point1, point2, sketch)

            create_pipe(feats, path, outDiameter=0.8, wallThickness=0.05)

        ui.messageBox("Pipes successfully created.")

    except:
        if ui:
            ui.messageBox("Failed:\n{}".format(traceback.format_exc()))
