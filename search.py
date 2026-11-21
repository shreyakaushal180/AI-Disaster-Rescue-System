from collections import deque
import heapq

def bfs(env, start, goal):
    if not env.is_valid(*start) or not env.is_valid(*goal):
        return None

    queue = deque([start])
    parents = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            return _build_path(parents, goal)

        for neighbor in env.get_neighbors(*current):
            if neighbor not in parents:
                parents[neighbor] = current
                queue.append(neighbor)

    return None

def dfs(env, start, goal):
    if not env.is_valid(*start) or not env.is_valid(*goal):
        return None

    stack = [start]
    parents = {start: None}

    while stack:
        current = stack.pop()
        if current == goal:
            return _build_path(parents, goal)

        for neighbor in env.get_neighbors(*current):
            if neighbor not in parents:
                parents[neighbor] = current
                stack.append(neighbor)

    return None

def _build_path(parents, goal):
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parents[current]
    return list(reversed(path))

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(env, start, goal):
    return _a_star(env, start, goal, risk_weight=0)

def a_star_risk(env, start, goal, risk_weight=3):
    return _a_star(env, start, goal, risk_weight)

def _a_star(env, start, goal, risk_weight):
    if risk_weight < 0:
        raise ValueError("risk_weight must be non-negative.")
    if not env.is_valid(*start) or not env.is_valid(*goal):
        return None

    queue = [(heuristic(start, goal), 0, start)]
    best_cost = {start: 0}
    parents = {start: None}

    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost != best_cost.get(current):
            continue
        if current == goal:
            return _build_path(parents, goal)

        for neighbor in env.get_neighbors(*current):
            step_cost = 1 + env.grid[neighbor[0]][neighbor[1]].risk_level * risk_weight
            next_cost = cost + step_cost
            if next_cost >= best_cost.get(neighbor, float("inf")):
                continue
            best_cost[neighbor] = next_cost
            parents[neighbor] = current
            priority = next_cost + heuristic(neighbor, goal)
            heapq.heappush(queue, (priority, next_cost, neighbor))

    return None

def greedy_best_first(env, start, goal):
    if not env.is_valid(*start) or not env.is_valid(*goal):
        return None

    queue = [(heuristic(start, goal), start)]
    parents = {start: None}

    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            return _build_path(parents, goal)
        for neighbor in env.get_neighbors(*current):
            if neighbor not in parents:
                parents[neighbor] = current
                heapq.heappush(queue, (heuristic(neighbor, goal), neighbor))
    return None

def hill_climbing(env, start, goal):
    if not env.is_valid(*start) or not env.is_valid(*goal):
        return None

    current = start
    path = [current]

    while current != goal:
        x, y = current

        neighbors = env.get_neighbors(x, y)

        if not neighbors:
            return path  # stuck

        # choose best neighbor based on heuristic
        best = None
        best_h = float('inf')

        for nx, ny in neighbors:
            h = heuristic((nx, ny), goal)
            if h < best_h:
                best_h = h
                best = (nx, ny)

        # if no improvement -> stuck
        if best_h >= heuristic(current, goal):
            print("Hill Climbing stuck at:", current)
            return path

        current = best
        path.append(current)

    return path