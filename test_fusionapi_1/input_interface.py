import yaml

def input_interface():
    points = []

    # Input points
    while True:
        print("\nEnter coordinates for the point:")
        x = float(input("X: "))
        y = float(input("Y: "))
        z = float(input("Z: "))

        is_linkage = input("Is it linkage (True/False): ").strip().lower()
        is_linkage = True if is_linkage in ['true', '1', 't', 'yes', 'y'] else False

        points.append({
            'coordinates': {'x': x, 'y': y, 'z': z},
            'is_linkage': is_linkage
        })

        finished = input("Have you finished input (True/False): ").strip().lower()
        if finished in ['true', '1', 't', 'yes', 'y']:
            break

    # Display points with numbers
    connections = []
    remaining_points = points.copy()

    while remaining_points:
        print("\nCurrent points:")
        for idx, pt in enumerate(remaining_points):
            print(f"{idx+1}: Coordinates={pt['coordinates']}, is_linkage={pt['is_linkage']}")

        first_idx = 0
        first_point = remaining_points[first_idx]
        connect_to = input(f"\nEnter point number(s) to connect with point {first_idx+1} (use comma to separate multiple points, e.g., 2,3): ")
        connect_indices = [int(x.strip()) for x in connect_to.split(',') if x.strip().isdigit()]
        connect_to_points = [remaining_points[x-1] for x in connect_indices]
        # Save paired points
        connections_points = [first_point]+[connect_to_point for connect_to_point in connect_to_points]
        connection_idx = [first_idx] + [i - 1 for i in connect_indices]
        connections.append(connections_points)

        # Remove paired points without changing original numbering
        for idx in sorted(connection_idx, reverse=True):
            remaining_points.pop(idx) #pop items in reverse order to avoid index error

    print("\nFinal paired points:")
    for idx, pair in enumerate(connections):
        print(f"{idx+1}: {pair}")

    # Optionally save to YAML for future use
    with open('paired_points.yaml', 'w') as file:
        yaml.dump(connections, file)

    print("\nPaired points saved to paired_points.yaml")

if __name__ == "__main__":
    input_interface()