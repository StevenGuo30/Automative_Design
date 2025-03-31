import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------- Geometry Utils ----------------
def segment_distance(p1, p2, q1, q2):
    p1, p2, q1, q2 = map(np.array, (p1, p2, q1, q2))
    u = p2 - p1
    v = q2 - q1
    w0 = p1 - q1
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)
    denom = a * c - b * b
    if denom == 0:
        sc, tc = 0.0, d / b if b != 0 else 0.0
    else:
        sc = (b * e - c * d) / denom
        tc = (a * e - b * d) / denom
    sc = np.clip(sc, 0, 1)
    tc = np.clip(tc, 0, 1)
    cp_p = p1 + sc * u
    cp_q = q1 + tc * v
    return np.linalg.norm(cp_p - cp_q)

def is_intersecting(new_edge, points, edges, pipe_d):
    a1, b1 = new_edge
    for a2, b2 in edges:
        if a1 == a2 or a1 == b2 or b1 == a2 or b1 == b2:
            continue
        dist = segment_distance(points[a1], points[b1], points[a2], points[b2])
        if dist < pipe_d:
            return True
    return False

# ---------------- Union-Find ----------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py
            return True
        return False

# ---------------- Obstacle-Avoiding Connection Algorithm ----------------
def generate_pipe_paths(point_dict, group_connections, pipe_d=1.5):
    base_points = [np.array(p) for p in point_dict.values()]
    name_to_index = {name: idx for idx, name in enumerate(point_dict.keys())}
    adjusted_points = base_points.copy()
    global_edges = []
    edge_info = []
    frames = []

    offset_directions = [
        np.array([dx, dy, dz])
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [0, 1, 2]
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    for group_idx, group in enumerate(group_connections):
        print(f"\nProcessing group {group_idx+1}: {group}")
        original_indices = [name_to_index[name] for name in group]
        sub_points = [adjusted_points[i] for i in original_indices]
        n = len(sub_points)
        uf = UnionFind(n)

        candidates = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(sub_points[i] - sub_points[j])
                candidates.append((dist, i, j))
        candidates.sort()

        temp_edges = []
        temp_info = []

        for _, i, j in candidates:
            gi, gj = original_indices[i], original_indices[j]
            if uf.find(i) == uf.find(j):
                continue
            if not is_intersecting((gi, gj), adjusted_points, global_edges + temp_edges, pipe_d):
                temp_edges.append((gi, gj))
                temp_info.append((gi, gj, False))
                uf.union(i, j)
            else:
                # Try all previous inserted points as bridge
                for mid_idx in range(len(base_points), len(adjusted_points)):
                    edge1 = (gi, mid_idx)
                    edge2 = (mid_idx, gj)
                    if (not is_intersecting(edge1, adjusted_points, global_edges + temp_edges, pipe_d) and
                        not is_intersecting(edge2, adjusted_points, global_edges + temp_edges, pipe_d)):
                        temp_edges.append(edge1)
                        temp_edges.append(edge2)
                        temp_info.append((edge1[0], edge1[1], True))
                        temp_info.append((edge2[0], edge2[1], True))
                        uf.union(i, j)
                        print(f"[Avoid] Used existing point {mid_idx} between {gi}-{gj}")
                        break
                else:
                    # Try inserting new midpoint if no existing works
                    pi, pj = adjusted_points[gi], adjusted_points[gj]
                    base_mid = (pi + pj) / 2
                    inserted = False
                    for direction in offset_directions:
                        # Adjust scale as needed to avoid intersection
                        for scale in [1, 1.5, 2, 2.5, 3]: 
                            offset = direction * pipe_d * scale
                            mid = base_mid + offset
                            mid_idx = len(adjusted_points)
                            adjusted_points.append(mid)
                            edge1 = (gi, mid_idx)
                            edge2 = (mid_idx, gj)
                            if (not is_intersecting(edge1, adjusted_points, global_edges + temp_edges, pipe_d) and
                                not is_intersecting(edge2, adjusted_points, global_edges + temp_edges, pipe_d)):
                                temp_edges.append(edge1)
                                temp_edges.append(edge2)
                                temp_info.append((edge1[0], edge1[1], True))
                                temp_info.append((edge2[0], edge2[1], True))
                                uf.union(i, j)
                                print(f"[Avoid] Inserted new point {mid_idx} between {gi}-{gj}")
                                inserted = True
                                break
                        if inserted:
                            break

        global_edges.extend(temp_edges)
        edge_info.extend(temp_info)
        frames.append((adjusted_points.copy(), global_edges.copy()))

    print("\n[Edge Summary]:")
    for s, e, avoided in edge_info:
        print(f"Edge ({s}, {e}) {'[avoided]' if avoided else '[direct]'}")

    return frames, adjusted_points, global_edges

# ---------------- Visualization ----------------
def visualize_pipe_animation(frames):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()
        pts, eds = frames[frame_idx]
        pts = np.array(pts)

        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='blue', s=50)
        for i, (x, y, z) in enumerate(pts):
            ax.text(x, y, z, f'{i}', fontsize=10, color='red')

        for (i, j) in eds:
            pi, pj = pts[i], pts[j]
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]], [pi[2], pj[2]], 'r-', linewidth=2)

        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)
        ax.set_zlim(-1, 3)
        ax.set_title(f"Step {frame_idx+1}/{len(frames)}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True)

    ani = FuncAnimation(fig, update, frames=len(frames), interval=1500, repeat=False)
    plt.tight_layout()
    plt.show()


# ---------------- Main Function ----------------
if __name__ == "__main__":
    # Example points and connections
    point_dict = {
        'A': [0, 0, 0],
        'B': [0, 1, 0],
        'C': [1, 1, 0],
        'D': [1, 0, 0],
        'E': [0.5, 1, 0.2]
    }

    group_connections = [['A', 'C', 'E'], ['B', 'D']]

    frames, points, edges = generate_pipe_paths(point_dict, group_connections, pipe_d=0.15)
    visualize_pipe_animation(frames)