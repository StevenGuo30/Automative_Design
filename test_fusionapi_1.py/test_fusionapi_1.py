"""This file acts as the main module for this script."""

import traceback
import adsk.core
import adsk.fusion

# import adsk.cam

# Initialize the global variables for the Application and UserInterface objects.
app = adsk.core.Application.get()
ui = app.userInterface


import adsk.core, adsk.fusion, traceback


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = app.activeProduct
        rootComp = design.rootComponent

        # Create sketches

        # Sketch 1 (XY plane)
        sketches = rootComp.sketches
        xyPlane = rootComp.xYConstructionPlane
        sketch1 = sketches.add(xyPlane)
        lines1 = sketch1.sketchCurves.sketchLines
        start1 = adsk.core.Point3D.create(0, 0, 0)
        end1 = adsk.core.Point3D.create(10, 0, 0)
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
