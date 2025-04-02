"""
This module provides functions to create custom planes, optimized pipe paths,
and hollow pipe features using the Fusion 360 API.
All units are in centimeters.
"""

import traceback
import adsk.core
import adsk.fusion


def create_custom_plane(rootComp, point1, point2, point3):
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
    middle = adsk.core.Point3D.create((start.x + end.x) / 2, (start.y + end.y) / 2, (start.z + end.z) / 2)
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
    pipeInput = pipeFeatures.createInput(path, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    pipeInput.sectionSize = adsk.core.ValueInput.createByReal(outDiameter)
    pipeInput.isHollow = True
    pipeInput.wallThickness = adsk.core.ValueInput.createByReal(wallThickness)
    pipeFeature = pipeFeatures.add(pipeInput)
    return pipeFeature


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

        feats = rootComp.features
        sketches = rootComp.sketches

        point1 = adsk.core.Point3D.create(0, 0, 0)
        point2 = adsk.core.Point3D.create(1, 0, 0)
        point3 = adsk.core.Point3D.create(0, 1, 1)
        start1 = adsk.core.Point3D.create(0, 0, 0)
        end1 = adsk.core.Point3D.create(5, 5, 5)
        start2 = adsk.core.Point3D.create(0, 0, 0)
        end2 = adsk.core.Point3D.create(2, 3, 3)

        customPlane = create_custom_plane(rootComp, point1, point2, point3)
        sketch1 = sketches.add(customPlane)
        path1 = create_pipe_path(rootComp, feats, start1, end1, sketch1)
        sketch2 = sketches.add(customPlane)
        path2 = create_pipe_path(rootComp, feats, start2, end2, sketch2)

        create_pipe(feats, path1, outDiameter=0.8, wallThickness=0.05)
        create_pipe(feats, path2, outDiameter=0.5, wallThickness=0.02)

        ui.messageBox("Pipes successfully created.")

    except:
        if ui:
            ui.messageBox("Failed:\n{}".format(traceback.format_exc()))
