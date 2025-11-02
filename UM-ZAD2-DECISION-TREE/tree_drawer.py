import matplotlib.pyplot as plt
import numpy as np


def _count_leaves(node):
    if node is None:
        return 0
    if node.value is not None:
        return 1
    return _count_leaves(node.left) + _count_leaves(node.right)

def _assign_positions(node, x_min, x_max, y, y_step, positions, depths):
    """
    Recursively assign positions (x,y) for nodes.
    We use leaf counts to space leaves evenly.
    x_min, x_max: horizontal interval for this subtree
    y: current y coordinate
    y_step: vertical spacing
    positions: dict mapping node -> (x,y)
    depths: dict mapping node -> depth (int)
    """
    if node is None:
        return
    if node.value is not None:
        # leaf: place at center of interval
        x = (x_min + x_max) / 2.0
        positions[node] = (x, y)
        depths[node] = int(round(y / y_step * -1))  # optional depth store
        return
    # compute left and right leaf counts to split interval proportionally
    left_leaves = _count_leaves(node.left)
    right_leaves = _count_leaves(node.right)
    total = left_leaves + right_leaves
    if total == 0:
        x = (x_min + x_max) / 2.0
        positions[node] = (x, y)
        depths[node] = int(round(y / y_step * -1))
        return
    # allocate sub-intervals
    left_width = (left_leaves / total) * (x_max - x_min)
    mid = x_min + left_width
    # assign current node x as midpoint of children's intervals
    # but ensure fallback in degenerate case
    left_min = x_min
    left_max = mid
    right_min = mid
    right_max = x_max
    # recurse children first
    _assign_positions(node.left, left_min, left_max, y - y_step, y_step, positions, depths)
    _assign_positions(node.right, right_min, right_max, y - y_step, y_step, positions, depths)
    # set this node x to average of child x positions (if they exist)
    child_xs = []
    if node.left in positions:
        child_xs.append(positions[node.left][0])
    if node.right in positions:
        child_xs.append(positions[node.right][0])
    if child_xs:
        x = sum(child_xs) / len(child_xs)
    else:
        x = (x_min + x_max) / 2.0
    positions[node] = (x, y)
    depths[node] = int(round(y / y_step * -1))


def draw_tree_png(root, filename="tree.png", figsize=(20,6), fontsize=10, y_step=0.12):
    """
    Draw the decision tree rooted at `root` and save it to `filename`.
    - feature_names: list of feature names (indexed by feature idx)
    - node_box_pad: horizontal pad for box widths (visual)
    - y_step: vertical distance between levels (fraction of figure height)
    """
    total_leaves = _count_leaves(root)
    if total_leaves == 0:
        raise ValueError("Tree has no leaves to draw.")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.set_axis_off()

    # We'll use a normalized coordinate system x in [0,1], y in [0,1]
    top_y = 0.95

    positions = {}
    depths = {}
    _assign_positions(root, 0.02, 0.98, top_y, y_step, positions, depths)

    # draw edges (parent->child)
    for node, (x, y) in positions.items():
        if node.value is not None:
            continue
        for child in (node.left, node.right):
            if child is None:
                continue
            if child not in positions:
                continue
            cx, cy = positions[child]
            ax.plot([x, cx], [y - 0.025, cy + 0.02], linewidth=0.9, color="black")

    # draw nodes (boxes + text)
    for node, (x, y) in positions.items():
        if node.value is not None:
            label = node.value
        else:
            feature = node.feature
            thresh = node.threshold
            if isinstance(node.threshold, (int, float, np.number)):
                label = f"{feature} <= {thresh}"
            else:
                label = f"{feature} == {thresh}"
        # draw text box
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", pad=0.4))

    plt.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved tree drawing to {filename}")
