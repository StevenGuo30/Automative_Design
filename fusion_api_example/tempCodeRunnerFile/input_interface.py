import os
import json
import string

script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
json_path = os.path.join(script_dir, "paired_points.json")  # 改为 JSON 路径


def name_generator():
    import string

    letters = string.ascii_uppercase
    count = 0
    while True:
        letter = letters[count % 26]
        suffix_index = count // 26
        if suffix_index == 0:
            yield letter
        else:
            yield f"{letter}_{suffix_index}"
        count += 1


def input_interface():
    points = []
    name_gen = name_generator()

    # Input points
    while True:
        print("\nAll input Units are in cm.")
        print("\nEnter coordinates for the point:(use comma to separate x,y,z)")
        print("Example: 1,2,3")
        coordinates = input("Coordinates: ")
        coords = [
            float(x.strip())
            for x in coordinates.split(",")
            if x.strip().replace(".", "", 1).isdigit()
        ]
        if len(coords) != 3:
            print("Invalid input. Please enter exactly three numbers.")
            continue
        x, y, z = coords

        # Input direction
        print("\nEnter direction for the point (use comma to separate x,y,z)")
        print("Example: 1,0,0")
        direction = input("Direction: ")
        direction = [
            float(x.strip())
            for x in direction.split(",")
            if x.strip().replace(".", "", 1).isdigit()
        ]
        if len(direction) != 3:
            print("Invalid input. Please enter exactly three numbers.")
            continue
        x_dir, y_dir, z_dir = direction

        is_linkage = input("Is it linkage (True/False): ").strip().lower()
        is_linkage = True if is_linkage in ["true", "1", "t", "yes", "y"] else False

        points.append(
            {
                "name": next(name_gen),
                "coordinates": {"x": x, "y": y, "z": z},
                "directions": {"x": x_dir, "y": y_dir, "z": z_dir},
                "is_linkage": is_linkage,
            }
        )

        finished = input("Have you finished input (True/False): ").strip().lower()
        if finished in ["true", "1", "t", "yes", "y"]:
            break

    # Display points with numbers
    connections = []
    remaining_points = points.copy()

    while remaining_points:
        print("\nCurrent points:")
        for idx, pt in enumerate(remaining_points):
            print(
                f"{idx+1}: Coordinates={pt['coordinates']}, Direction={pt['directions']}, is_linkage={pt['is_linkage']}"
            )

        first_idx = 0
        first_point = remaining_points[first_idx]
        connect_to = input(
            f"\nEnter point number(s) to connect with point {first_idx+1} (use comma to separate multiple points, e.g., 2,3): "
        )
        connect_indices = [
            int(x.strip()) for x in connect_to.split(",") if x.strip().isdigit()
        ]
        connect_to_points = [remaining_points[x - 1] for x in connect_indices]
        # Save paired points
        connections_points = [first_point] + [
            connect_to_point for connect_to_point in connect_to_points
        ]
        connection_idx = [first_idx] + [i - 1 for i in connect_indices]
        connections.append(connections_points)

        # Remove paired points without changing original numbering
        for idx in sorted(connection_idx, reverse=True):
            remaining_points.pop(idx)  # pop items in reverse order to avoid index error

    print("\nFinal paired points:")
    for idx, pair in enumerate(connections):
        print(f"{idx+1}: {pair}")

    # ✅ 保存为 JSON 格式
    with open(json_path, "w") as file:
        json.dump(connections, file, indent=2)

    print(f"\nPaired points saved to {json_path}")


if __name__ == "__main__":
    input_interface()
