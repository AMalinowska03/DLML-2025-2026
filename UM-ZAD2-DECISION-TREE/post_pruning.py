import numpy as np

def prune_tree(tree, X_val, y_val):

    def _predict_single(node, x):
        while node.value is None:
            val = x[node.feature]
            if isinstance(val, list):
                go_left = node.threshold in val
            else:
                if isinstance(node.threshold, (int, float, np.number)):
                    go_left = val <= node.threshold
                else:
                    go_left = val == node.threshold
            node = node.left if go_left else node.right
        return node.value

    def _predict_all(node, X):
        return np.array([_predict_single(node, x) for x in X])

    def _prune_node(node):
        # Recurse first (bottom-up)
        if node.left and node.left.value is None:
            _prune_node(node.left)
        if node.right and node.right.value is None:
            _prune_node(node.right)

        # if both children are leaves -> consider pruning this node
        if node.left and node.left.value is not None and node.right and node.right.value is not None:
            # current performance
            y_pred_before = _predict_all(tree.root, X_val)
            acc_before = np.mean(y_pred_before == y_val)

            # store current children
            left, right = node.left, node.right

            # replace this node with a leaf (majority label)
            subtree_labels = _predict_all(node, X_val)
            majority_label = np.bincount(subtree_labels.astype(int)).argmax()
            node.value = majority_label
            node.left = None
            node.right = None

            y_pred_after = _predict_all(tree.root, X_val)
            acc_after = np.mean(y_pred_after == y_val)

            # if pruning hurt accuracy, revert
            if acc_after < acc_before:
                node.value = None
                node.left = left
                node.right = right

    _prune_node(tree.root)
