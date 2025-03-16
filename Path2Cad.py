import adsk.core, adsk.fusion, adsk.cam, traceback


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = app.activeProduct

        rootComp = design.rootComponent
        sketches = rootComp.sketches
        xyPlane = rootComp.xYConstructionPlane
        sketch = sketches.add(xyPlane)

        lines = sketch.sketchCurves.sketchLines
        startPoint = adsk.core.Point3D.create(0, 0, 0)
        endPoint = adsk.core.Point3D.create(10, 0, 0)
        lines.addByTwoPoints(startPoint, endPoint)

        ui.messageBox("Line created successfully.")
    except:
        if ui:
            ui.messageBox("Failed:\n{}".format(traceback.format_exc()))
