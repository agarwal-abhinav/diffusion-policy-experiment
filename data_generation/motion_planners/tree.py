class TreeNode:
    def __init__(self, value, parent = None):
        assert isinstance(parent, TreeNode) or parent is None
        self.value = value
        self.parent = parent
        self.children = []

class Tree:
    def __init__(self, root: TreeNode):
        self.root = root

    def add_node(self, node: TreeNode, parent: TreeNode):
        parent.children.append(node)
        node.parent = parent

    def find_path_to_root(self, node: TreeNode):
        path = []
        dummy_node = TreeNode(node.value, node.parent) # dummy node to avoid modifying the tree
        while dummy_node is not None:
            path.append(dummy_node.value)
            dummy_node = dummy_node.parent
        path.reverse()
        return path