"""
Directed graph utilities for acyclicity checking and topological sorting.
Used by Algorithms 1 and 2 to maintain and check the arc set A_t.
"""


def is_acyclic(arc_set: set, d: int) -> bool:
    """Check if directed graph ([d], arc_set) is acyclic via DFS."""
    adj = {i: [] for i in range(d)}
    for i, j in arc_set:
        adj[i].append(j)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * d

    def dfs(u: int) -> bool:
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:
                return False
            if color[v] == WHITE and not dfs(v):
                return False
        color[u] = BLACK
        return True

    for u in range(d):
        if color[u] == WHITE and not dfs(u):
            return False
    return True


def topological_sort(arc_set: set, d: int) -> list:
    """Return topological ordering of [d] given arc_set.

    Returns a list of nodes such that for every arc (i,j), i comes before j.
    Returns empty list if graph has a cycle.
    """
    adj = {i: [] for i in range(d)}
    in_deg = [0] * d
    for i, j in arc_set:
        adj[i].append(j)
        in_deg[j] += 1

    queue = [i for i in range(d) if in_deg[i] == 0]
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in adj[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    if len(order) != d:
        return []  # Cycle detected
    return order
