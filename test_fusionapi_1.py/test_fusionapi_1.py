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

        # Create custom plane
        try:
            customPlane = create_custom_plane(rootComp, point1, point2, point3)
        except:
            ui.messageBox("Failed to create custom plane.")
        # Create sketches

        # Sketch 1 (XY plane)
        sketches = rootComp.sketches
        xyPlane = rootComp.xYConstructionPlane
        sketch1 = sketches.add(customPlane)
        lines1 = sketch1.sketchCurves.sketchLines
        start1 = adsk.core.Point3D.create(
            0, 0, 0
        )  # Here is the local cordiantes of the sketch
        end1 = adsk.core.Point3D.create(0, 15, 0)
        line1 = lines1.addByTwoPoints(start1, end1)

        # Sketch 2 (XY plane)
        sketch2 = sketches.add(xyPlane)
        lines2 = sketch2.sketchCurves.sketchLines
        start2 = adsk.core.Point3D.create(0, 5, 0)
        mid2 = adsk.core.Point3D.create(5, 10, 0)
        end2 = adsk.core.Point3D.create(10, 5, 0)
        line2_1 = lines2.addByTwoPoints(start2, mid2)
        line2_2 = lines2.addByTwoPoints(mid2, end2)

        # Create paths
        feats = rootComp.features
        path1 = feats.createPath(line1)

        # For path2, use ObjectCollection to include multiple lines
        lineCollection = adsk.core.ObjectCollection.create()
        lineCollection.add(line2_1)
        lineCollection.add(line2_2)
        path2 = feats.createPath(lineCollection)

        # Pipe feature
        pipes = feats.pipeFeatures

        # Set pipe diameter
        pipeRadius = adsk.core.ValueInput.createByReal(0.5)  # Radius = 0.5 cm

        # Create pipe 1
        pipeInput1 = pipes.createInput(
            path1, adsk.fusion.FeatureOperations.NewBodyFeatureOperation
        )
        pipeInput1.sectionRadius = pipeRadius
        pipe1 = pipes.add(pipeInput1)

        # Create pipe 2
        pipeInput2 = pipes.createInput(
            path2, adsk.fusion.FeatureOperations.NewBodyFeatureOperation
        )
        pipeInput2.sectionRadius = pipeRadius
        pipe2 = pipes.add(pipeInput2)

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
