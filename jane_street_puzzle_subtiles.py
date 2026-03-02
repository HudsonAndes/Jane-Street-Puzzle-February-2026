# =====================================================
#
#    █████╗  ██████╗
#   ██╔══██╗██╔════╝
#   ███████║██║  ███╗
#   ██╔══██║██║   ██║
#   ██║  ██║╚██████╔╝
#   ╚═╝  ╚═╝ ╚═════╝
#
#   Created by Angel Guerrero
#
# =====================================================

from ortools.sat.python import cp_model
from collections import defaultdict
import time
import sys
import os
import concurrent.futures

# =====================================================
# CONFIG & CLUES
# =====================================================

GRID = 13
MAX_K = 16

CLUES_RAW = [
    (0,4,15),(1,7,11),
    (2,1,15),(2,3,5),(2,6,15),(2,8,11),(2,10,11),
    (3,4,15),(3,7,8),(3,9,12),(3,11,12),
    (4,1,16),(4,5,8),(4,10,6),
    (5,3,16),(5,12,6),
    (6,2,16),(6,4,3),(6,6,16),(6,8,1),(6,10,12),
    (7,0,13),(7,9,4),
    (8,2,7),(8,7,12),(8,11,10),
    (9,1,2),(9,3,13),(9,5,16),(9,8,14),
    (10,2,13),(10,4,14),(10,6,14),(10,9,14),(10,11,10),
    (11,5,9),(12,8,9),
]

# k -> set of (r,c) grid positions required for that value
CLUES = defaultdict(set)
for r, c, v in CLUES_RAW:
    CLUES[v].add((r, c))

# Full clue location map — used to block invalid placements
ALL_CLUE_LOCS = {(r,c): v for r,c,v in CLUES_RAW}

# =====================================================
# SHAPE LOGIC
# =====================================================

def normalize(cells):
    """Shift shape to origin and sort — canonical form for deduplication."""
    if not cells: return tuple()
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))

def get_transforms(shape):
    """All unique rotations and reflections of a shape."""
    transforms = set()
    current = list(shape)
    for _ in range(4):
        current = [(c, -r) for r, c in current]
        transforms.add(normalize(current))
        transforms.add(normalize([(-r, c) for r, c in current]))
    return transforms

def compatible_with_clues(shape, k):
    """
    Can this shape (size k) cover all required clue positions for k?
    Tests every transform — returns True on first valid alignment.
    """
    required_cells = CLUES[k]
    if not required_cells:
        return True

    req_list = list(required_cells)

    for t_shape in get_transforms(shape):
        pivot_r, pivot_c = req_list[0]

        for (sr, sc) in t_shape:
            dr = pivot_r - sr
            dc = pivot_c - sc

            valid = True
            for (rr, rc) in req_list:
                if (rr - dr, rc - dc) not in t_shape:
                    valid = False
                    break

            if valid:
                return True

    return False

# =====================================================
# SHAPE GENERATION (PARALLEL)
# =====================================================

def worker_generate_batch(args):
    """
    Worker: grows a batch of K-1 shapes to K, filters against clues.
    Returns candidates and their parent mappings.
    """
    k, parents = args
    local_candidates = set()
    local_parent_map = defaultdict(set)

    for parent in parents:
        parent_set = set(parent)

        neighbors = set()
        for r, c in parent:
            for nr, nc in [(r+1,c), (r-1,c), (r,c+1), (r,c-1)]:
                if (nr, nc) not in parent_set:
                    neighbors.add((nr, nc))

        for nr, nc in neighbors:
            new_shape = normalize(parent + ((nr, nc),))

            if not compatible_with_clues(new_shape, k):
                continue

            local_candidates.add(new_shape)
            local_parent_map[new_shape].add(parent)

    return local_candidates, local_parent_map

def generate_shapes():
    print("Step 1: Generating Valid Shapes (Parallel)")
    print("-" * 50)

    valid_shapes = {1: {((0,0),)}}
    parent_map = defaultdict(lambda: defaultdict(set))

    workers = min(32, os.cpu_count() or 1)
    print(f"  Workers: {workers}")

    for k in range(2, MAX_K + 1):
        t0 = time.time()

        parents = list(valid_shapes[k-1])
        if not parents:
            print(f"Error: No valid shapes at K={k}")
            return None, None

        chunk_size = max(1, len(parents) // (workers * 4))
        batches = [parents[i:i + chunk_size] for i in range(0, len(parents), chunk_size)]

        candidates = set()

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            tasks = [(k, batch) for batch in batches]
            results = executor.map(worker_generate_batch, tasks)

            for local_cand, local_pmap in results:
                candidates.update(local_cand)
                for child, ps in local_pmap.items():
                    parent_map[k][child].update(ps)

        valid_shapes[k] = candidates
        print(f"  K={k:2d}: {len(candidates):8d} shapes ({time.time()-t0:.2f}s)")

        if not candidates:
            print(f"No shapes found at K={k} — stopping.")
            return None, None

    # Backward pruning — drop K-1 shapes that dead-end at K
    print("\nStep 2: Backward Pruning")
    print("-" * 50)
    for k in range(MAX_K, 1, -1):
        needed = set()
        for shape in valid_shapes[k]:
            needed.update(parent_map[k][shape])

        before = len(valid_shapes[k-1])
        valid_shapes[k-1] = valid_shapes[k-1].intersection(needed)
        after = len(valid_shapes[k-1])

        print(f"  K={k-1}: {before} -> {after}")

    return valid_shapes, parent_map

# =====================================================
# PLACEMENT GENERATION
# =====================================================

def generate_placements(k, shapes):
    """
    All valid grid placements for value K.
    Each placement must cover required clues and avoid other clue positions.
    """
    placements = []
    required = CLUES[k]
    forbidden = {loc for loc, val in ALL_CLUE_LOCS.items() if val != k}

    for shape in shapes:
        for t_shape in get_transforms(shape):
            max_r = max(r for r, c in t_shape)
            max_c = max(c for r, c in t_shape)

            for dr in range(GRID - max_r):
                for dc in range(GRID - max_c):
                    cells = set((r+dr, c+dc) for r, c in t_shape)

                    if not required.issubset(cells): continue
                    if not cells.isdisjoint(forbidden): continue

                    placements.append((tuple(sorted(cells)), shape))

    return placements

# =====================================================
# VERIFICATION
# =====================================================

def is_connected(cells):
    """BFS connectivity check — all cells must form one orthogonal region."""
    if not cells:
        return True
    cells_set = set(cells)
    start = next(iter(cells_set))
    visited = set()
    stack = [start]
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        for nr, nc in [(r+1,c), (r-1,c), (r,c+1), (r,c-1)]:
            if (nr, nc) in cells_set and (nr, nc) not in visited:
                stack.append((nr, nc))
    return len(visited) == len(cells_set)

def verify_solution_strict(grid):
    """Full solution verification: cell counts, connectivity, clues, and containment."""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    errors = []

    # Cell count per K
    print("\n[1] Cell counts...")
    for k in range(1, MAX_K + 1):
        cells = [(r,c) for r in range(GRID) for c in range(GRID) if grid[r][c] == k]
        if len(cells) != k:
            errors.append(f"K={k}: {len(cells)} cells (expected {k})")
        else:
            print(f"  K={k:2d}: OK")

    # Connectivity
    print("\n[2] Connectivity...")
    for k in range(1, MAX_K + 1):
        cells = {(r,c) for r in range(GRID) for c in range(GRID) if grid[r][c] == k}
        if cells and not is_connected(cells):
            errors.append(f"K={k}: disconnected region")
        else:
            print(f"  K={k:2d}: connected")

    # Clues
    print("\n[3] Clues...")
    clues_ok = True
    for (r, c), v in ALL_CLUE_LOCS.items():
        if grid[r][c] != v:
            errors.append(f"Clue ({r},{c}): expected {v}, got {grid[r][c]}")
            clues_ok = False
    if clues_ok:
        print(f"  All {len(ALL_CLUE_LOCS)} clues valid.")

    # Shape containment: Shape(K) must contain Shape(K-1)
    print("\n[4] Containment...")
    for k in range(2, MAX_K + 1):
        cells_k   = [(r,c) for r in range(GRID) for c in range(GRID) if grid[r][c] == k]
        cells_km1 = [(r,c) for r in range(GRID) for c in range(GRID) if grid[r][c] == k-1]

        shape_k   = normalize(cells_k)
        shape_km1 = normalize(cells_km1)

        found = False
        for t_km1 in get_transforms(shape_km1):
            t_km1_set = set(t_km1)
            for t_k in get_transforms(shape_k):
                t_k_set = set(t_k)
                for anchor in t_k:
                    for km1_anchor in t_km1:
                        dr = anchor[0] - km1_anchor[0]
                        dc = anchor[1] - km1_anchor[1]
                        shifted = {(r+dr, c+dc) for r, c in t_km1_set}
                        if shifted.issubset(t_k_set):
                            found = True
                            break
                    if found: break
                if found: break
            if found: break

        if not found:
            errors.append(f"K={k} does not contain K={k-1}")
        else:
            print(f"  K={k:2d} ⊃ K={k-1:2d}: OK")

    print("\n" + "-"*60)
    if errors:
        print("Verification failed:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("All checks passed.")
        return True

# =====================================================
# SOLVER
# =====================================================

def solve():
    t_start = time.time()
    shapes_by_k, parent_map = generate_shapes()

    if not shapes_by_k:
        return

    print("\nStep 3: Building CP-SAT Model")
    print("-" * 50)
    model = cp_model.CpModel()

    placements_by_k = {}
    x_vars = {}
    shape_vars = {}

    # One placement per K — exact cover
    for k in range(1, MAX_K + 1):
        placements = generate_placements(k, shapes_by_k[k])
        placements_by_k[k] = placements
        print(f"  K={k}: {len(placements)} placements")

        k_vars = []
        for i in range(len(placements)):
            v = model.NewBoolVar(f"x_{k}_{i}")
            x_vars[(k, i)] = v
            k_vars.append(v)
        model.AddExactlyOne(k_vars)

        placements_for_shape = defaultdict(list)
        for i, (cells, shape) in enumerate(placements):
            placements_for_shape[shape].append(i)

        for shape, idx_list in placements_for_shape.items():
            s_var = model.NewBoolVar(f"shape_{k}_{hash(shape)}")
            shape_vars[(k, shape)] = s_var

            model.Add(sum(x_vars[(k,i)] for i in idx_list) == 1).OnlyEnforceIf(s_var)
            model.Add(sum(x_vars[(k,i)] for i in idx_list) == 0).OnlyEnforceIf(s_var.Not())

    # No overlapping cells
    cell_usage = defaultdict(list)
    for k in range(1, MAX_K + 1):
        for i, (cells, shape) in enumerate(placements_by_k[k]):
            for r, c in cells:
                cell_usage[(r,c)].append(x_vars[(k,i)])

    for cell, vars_list in cell_usage.items():
        if len(vars_list) > 1:
            model.AddAtMostOne(vars_list)

    # Shape(K) must have a valid Shape(K-1) parent
    for k in range(2, MAX_K + 1):
        for shape in shapes_by_k[k]:
            if (k, shape) not in shape_vars: continue

            s_var = shape_vars[(k, shape)]
            parents = parent_map[k][shape]

            valid_parent_vars = [
                shape_vars[(k-1, p)]
                for p in parents
                if (k-1, p) in shape_vars
            ]

            if not valid_parent_vars:
                model.Add(s_var == 0)
            else:
                model.Add(sum(valid_parent_vars) >= 1).OnlyEnforceIf(s_var)

    print(f"Model ready. Solving... ({time.time() - t_start:.2f}s elapsed)")

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 8
    solver.parameters.max_time_in_seconds = 3600

    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("\nSolution found.")
        final_grid = [[0]*GRID for _ in range(GRID)]

        for k in range(1, MAX_K + 1):
            for i, (cells, shape) in enumerate(placements_by_k[k]):
                if solver.Value(x_vars[(k,i)]):
                    for r, c in cells:
                        final_grid[r][c] = k

        if verify_solution_strict(final_grid):
            print_grid(final_grid)
            compute_answer(final_grid)
        else:
            print("\nSolution found but failed verification.")
            print_grid(final_grid)
    else:
        print("No solution found.")

def print_grid(grid):
    print("\n" + "+" + "----+" * GRID)
    for r in range(GRID):
        row = "|"
        for c in range(GRID):
            v = grid[r][c]
            tag = "*" if (r,c) in ALL_CLUE_LOCS else " "
            if v == 0:
                row += "    |"
            else:
                row += f"{v:>2d}{tag}|"
        print(row)
        print("+" + "----+" * GRID)

def compute_answer(grid):
    print("\n" + "="*60)
    print("ANSWER")
    print("="*60)

    print("\nRow sums:")
    full_sums = []
    for r in range(GRID):
        row_total = sum(grid[r])
        full_sums.append(row_total)
        cells_detail = [(c, grid[r][c]) for c in range(GRID) if grid[r][c] > 0]
        clue_sum = sum(grid[r][c] for c in range(GRID) if (r,c) in ALL_CLUE_LOCS)
        print(f"  Row {r:2d}: {row_total:4d}  (Clues: {clue_sum:3d})  Cells: {len(cells_detail)}")

    mn = min(full_sums)
    mx = max(full_sums)

    print(f"\n" + "-"*60)
    print(f"Min row sum: {mn}")
    print(f"Max row sum: {mx}")
    print(f"\n{'='*60}")
    print(f"Answer: {mn} * {mx} = {mn * mx}")
    print(f"{'='*60}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    solve()