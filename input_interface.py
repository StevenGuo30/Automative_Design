import os
import json
import string
import numpy as np
import itertools

script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "paired_points.json")

def name_generator():
    letters = string.ascii_uppercase
    count = 0
    while True:
        letter = letters[count % 26]
        suffix = count // 26
        yield letter if suffix == 0 else f"{letter}_{suffix}"
        count += 1

def linkage_boxes_intersect(p1, p2):
    """
    Check if two linkage units (as oriented boxes) would interfere in space.
    Approximates each linkage as an OBB (oriented bounding box).
    The boxes are defined by their center coordinates(points are at the bottom of the box) and direction vectors(define the long side).
    """
    LENGTH = 2.1845-0.001
    WIDTH = HEIGHT = 1.7912-0.001

    c1 = np.array(list(p1["coordinates"].values()))
    c2 = np.array(list(p2["coordinates"].values()))

    d1 = np.array(list(p1["directions"].values()))
    d2 = np.array(list(p2["directions"].values()))
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)

    def get_obb_axes(d):
        v = np.array([1, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1, 0])
        side = np.cross(d, v)
        side /= np.linalg.norm(side)
        up = np.cross(d, side)
        return d, side, up

    def get_corners(center, axes):
        l, w, h = LENGTH / 2, WIDTH / 2, HEIGHT / 2
        corners = []
        for dx in [-l, l]:
            for dy in [-w, w]:
                for dz in [-h, h]:
                    corner = center + dx * axes[0] + dy * axes[1] + dz * axes[2]
                    corners.append(corner)
        return np.array(corners)

    axes1 = get_obb_axes(d1)
    axes2 = get_obb_axes(d2)
    c1 = c1 + (HEIGHT / 2) * axes1[2]  # Here we change the center of the box to the bottom
    corners1 = get_corners(c1, axes1) 
    c2 = c2 + (HEIGHT / 2) * axes2[2]  # Here we change the center of the box to the bottom
    corners2 = get_corners(c2, axes2)

    def sat_test(axes):
        for axis in axes:
            axis = axis / np.linalg.norm(axis)
            proj1 = [np.dot(c, axis) for c in corners1]
            proj2 = [np.dot(c, axis) for c in corners2]
            if max(proj1) < min(proj2) or max(proj2) < min(proj1):
                return False  # Separated
        return True  # No separating axis → intersect

    # Combine all test axes
    test_axes = list(axes1 + axes2)
    return sat_test(test_axes)


def check_compliance(points):
    linkage_points = [p for p in points if p["is_linkage"]]

    for a, b in itertools.combinations(linkage_points, 2):
        if linkage_boxes_intersect(a, b):
            print(f"Compliance failed: {a['name']} & {b['name']} interfere in space.")
            return [a, b]

    print("All linkage placements are compliant.")
    return []



def get_point_input(name):
    print(f"\nEnter data for point {name}:")
    coords = get_float_vector("Coordinates (x,y,z): ", 3)
    dirs = get_float_vector("Direction (x,y,z): ", 3) # Here the direction is where the thread is going
    is_linkage = input("Is it linkage (True/False): ").strip().lower() in ["true", "1", "yes", "y"]
    return {
        "name": name,
        "coordinates": dict(zip(["x", "y", "z"], coords)),
        "directions": dict(zip(["x", "y", "z"], dirs)),
        "is_linkage": is_linkage,
    }


def get_float_vector(prompt, length):
    while True:
        try:
            values = [float(x.strip()) for x in input(prompt).split(",")]
            if len(values) == length:
                return values
        except ValueError:
            pass
        print(f"Invalid input. Enter exactly {length} numeric values.")


def input_interface():
    points = []
    name_gen = name_generator()

    while True:
        points.append(get_point_input(next(name_gen)))
        if input("Finished input? (True/False): ").strip().lower() in ["true", "1", "yes", "y"]:
            break

    while True:
        errors = check_compliance(points)
        if not errors:
            break
        for error in errors:
            print(f"Re-enter data for point {error['name']}:")
            points = [p for p in points if p["name"] != error["name"]]
            points.append(get_point_input(error["name"]))

    print("\nAll points after compliance:")
    for i, pt in enumerate(points, 1):
        print(f"{i}: {pt}")

    connections = []
    remaining = points.copy()

    while remaining:
        base = remaining[0]
        indices = input(f"\nConnect point 1 ({base['name']}) with (e.g., 2,3): ")
        try:
            targets = [int(i.strip()) - 1 for i in indices.split(",")]
            if any(i <= 0 or i >= len(remaining) for i in targets) or 0 in targets:
                raise ValueError
        except ValueError:
            print("Invalid selection.")
            continue

        group = [base] + [remaining[i] for i in targets]
        connections.append(group)
        for i in sorted([0] + targets, reverse=True):
            remaining.pop(i)

    pipe_radius = float(input("\nEnter pipe radius (cm): "))
    output = {"pipe_radius": pipe_radius, "connections": connections}

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nData saved to {json_path}")


if __name__ == "__main__":
    input_interface()
