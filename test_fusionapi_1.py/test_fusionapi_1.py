"""This file acts as the main module for this script."""

import traceback
import adsk.core
import adsk.fusion

# import adsk.cam

# Initialize the global variables for the Application and UserInterface objects.
app = adsk.core.Application.get()
ui = app.userInterface

# Every time running this script, it will only add new features to the existing design.
# If you want to start a new design, please create a new design and run this script again.


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = app.activeProduct
        rootComp = design.rootComponent

        # Define three points to create a custom plane
        point1 = adsk.core.Point3D.create(0, 0, 0)
        point2 = adsk.core.Point3D.create(1, 0, 0)
        point3 = adsk.core.Point3D.create(0, 1, 1)
        
        # Define start and end points for pipe path
        # TODO: Needs to make a interface for entering start and end points
        start1 = adsk.core.Point3D.create(0, 0, 0)
        end1 = adsk.core.Point3D.create(0,1,1)

        # Create custom plane
        try:
            customPlane = create_custom_plane(rootComp, point1, point2, point3)
        except:
            ui.messageBox("Failed to create custom plane.")

        # Create sketches
        # Sketch 1 (custom plane)
        sketches = rootComp.sketches
        sketch1 = sketches.add(customPlane)
        spline1 = create_pipe_path(rootComp, start1, end1, sketch1)

        # Create paths
        feats = rootComp.features
        path1 = feats.createPath(spline1)

        # Pipe feature
        pipes = feats.pipeFeatures

        # Set pipe diameter
        pipeRadius = adsk.core.ValueInput.createByReal(1.5)  # Radius = 1.5 mm

        # Create pipe 1
        pipeInput1 = pipes.createInput(
            path1, adsk.fusion.FeatureOperations.NewBodyFeatureOperation
        )
        pipeInput1.sectionRadius = pipeRadius
        pipe1 = pipes.add(pipeInput1)

        ui.messageBox("Pipes successfully created.")

    except:
        if ui:
            ui.messageBox("Failed:\n{}".format(traceback.format_exc()))


# Define sketch plane using 3 points (used to create pipe sketch plane to connect air channels)
def create_custom_plane(rootComp, point1, point2, point3):
    # Create a new sketch on the XY plane
    sketches = rootComp.sketches
    xyPlane = rootComp.xYConstructionPlane
    sketch = sketches.add(xyPlane)

    # Add sketch points at the specified coordinates
    sketchPoint1 = sketch.sketchPoints.add(point1)
    sketchPoint2 = sketch.sketchPoints.add(point2)
    sketchPoint3 = sketch.sketchPoints.add(point3)

    # Create the construction plane through the three sketch points
    planes = rootComp.constructionPlanes
    planeInput = planes.createInput()
    planeInput.setByThreePoints(sketchPoint1, sketchPoint2, sketchPoint3)
    customPlane = planes.add(planeInput)

    return customPlane


# Create main body (used to creat main hexagonal body to connect modular part so that pipe can penetrate through it)
# TODO: Needs to define connected hexagonal body. The distance between to hexagonal surface is fixed? Defined by the path of pipe?

# Create pipe path (used to create path for pipe to follow)
# Use global_to_local function to convert global coordinates to local coordinates
def global_to_local(global_point, sketch):
    # Get the parent component of the sketch
    parent_component = sketch.parentComponent

    # Get the normal vector of the sketch reference plane
    normal = sketch.referencePlane.geometry.normal

    # Get the origin of the sketch reference plane
    origin = sketch.referencePlane.geometry.origin

    # Calculate the sketch X and Y directions
    x_direction = sketch.xDirection
    y_direction = normal.crossProduct(x_direction)

    # Create a transformation matrix from global to sketch local coordinates
    transform = adsk.core.Matrix3D.create()
    transform.setWithCoordinateSystem(origin, x_direction, y_direction, normal)

    # Compute the inverse transformation matrix
    inverse_transform = transform.copy()
    inverse_transform.invert()

    # Transform global point to local coordinates
    local_point = global_point.copy()
    local_point.transformBy(inverse_transform)

    return local_point

# TODO: Needs to use some diagram to optimize the best pipe path
def get_optimized_pipe(start, end):
    points = adsk.core.ObjectCollection.create()
    points.add(start)
    #for now just add one middle point for testing
    middle1 = adsk.core.Point3D.create((start.x + end.x) / 2, (start.y + end.y) / 2, (start.z + end.z) / 2)
    points.add(middle1) 
    points.add(end)
    return points

def create_pipe_path(rootComp, start, end, sketch):
    start_local = global_to_local(start, sketch)
    end_local = global_to_local(end, sketch)
    
    # Get the points to form the spline path
    points = adsk.core.ObjectCollection.create()
    points = get_optimized_pipe(start_local, end_local)
    spline = sketch.sketchCurves.sketchFittedSplines.add(points)
    
    return spline