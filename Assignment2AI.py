import heapq
import random
import time

# --- 1. Goal State ---
GOAL = (1, 2, 3,
        8, 0, 4,
        7, 6, 5)

# --- 2. Utility: Print puzzle nicely ---
def print_puzzle(state):
    for i in range(0, 9, 3):
        print(" ".join(str(x) if x != 0 else " " for x in state[i:i+3]))
    print()

# --- 3. Get successors (valid moves) ---
def get_moves(state):
    moves = []
    i = state.index(0)
    r, c = i // 3, i % 3
    directions = [(-1,0,'Up'), (1,0,'Down'), (0,-1,'Left'), (0,1,'Right')]

    for dr, dc, action in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            new_state = list(state)
            j = nr * 3 + nc
            new_state[i], new_state[j] = new_state[j], new_state[i]
            moves.append((action, tuple(new_state)))
    return moves

# --- 4. Heuristics ---
def h_manhattan(state):
    dist = 0
    for i, tile in enumerate(state):
        if tile != 0:
            goal_i = GOAL.index(tile)
            dist += abs(i//3 - goal_i//3) + abs(i%3 - goal_i%3)
    return dist

def h_zero(state):
    return 0  # For Uniform Cost Search

# --- 5. Solvability check ---
def count_inversions(state):
    nums = [n for n in state if n != 0]
    inv = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] > nums[j]:
                inv += 1
    return inv

def is_solvable(state):
    return count_inversions(state) % 2 == count_inversions(GOAL) % 2

def random_solvable_state():
    while True:
        s = list(range(9))
        random.shuffle(s)
        if is_solvable(s):
            return tuple(s)

# --- 6. Generic A* or UCS solver ---
def solve(start, heuristic):
    frontier = [(heuristic(start), 0, start, [])]  # (f, g, state, path)
    visited = set()
    nodes = 0
    start_time = time.time()

    while frontier:
        f, g, state, path = heapq.heappop(frontier)
        if state in visited:
            continue
        visited.add(state)
        nodes += 1

        if state == GOAL:
            end_time = time.time()
            return path, nodes, end_time - start_time

        for action, new_state in get_moves(state):
            if new_state not in visited:
                new_g = g + 1
                new_f = new_g + heuristic(new_state)
                heapq.heappush(frontier, (new_f, new_g, new_state, path + [action]))

    return None, nodes, time.time() - start_time

# --- 7. Main Execution ---
if __name__ == "__main__":
    start = random_solvable_state()
    print("\n--- 8-Puzzle Solver ---\n")
    print("Goal State:")
    print_puzzle(GOAL)
    print("Start State:")
    print_puzzle(start)

    # Run UCS
    print("Running Uniform Cost Search (UCS)...")
    ucs_path, ucs_nodes, ucs_time = solve(start, h_zero)
    print(f"UCS -> Moves: {len(ucs_path)}, Nodes: {ucs_nodes}, Time: {ucs_time:.3f}s\n")

    # Run A*
    print("Running A* (Manhattan Distance)...")
    astar_path, astar_nodes, astar_time = solve(start, h_manhattan)
    print(f"A*  -> Moves: {len(astar_path)}, Nodes: {astar_nodes}, Time: {astar_time:.3f}s\n")

    # Compare
    print("--- Comparison ---")
    print(f"UCS explored {ucs_nodes} nodes; A* explored {astar_nodes} nodes.")
    if astar_nodes < ucs_nodes:
        print("A* was more efficient because it used a heuristic.")