import fipy as fp
import numpy as np

def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    k: float = 400.0,
    r_outer: float = None,
    h: float = 1_000.0,
    T_inf: float = 0.0,
    r_inner: float = None,   # <-- allow explicit r_inner
):
    # Use the provided r_inner and r_outer, or infer from mesh (with warning!)
    if r_inner is None:
        r_cells = mesh.cellCenters[0].value
        print(f"Cell centers: min={r_cells.min()}, max={r_cells.max()}")
        r_inner = float(r_cells.min())
        print(f"[WARNING] r_inner inferred from mesh: {r_inner:.4f}")
    if r_outer is None:
        r_cells = mesh.cellCenters[0].value
        print(f"Cell centers: min={r_cells.min()}, max={r_cells.max()}")
        r_outer = float(r_cells.max())
        print(f"[WARNING] r_outer inferred from mesh: {r_outer:.4f}")

    print(f"[DEBUG] mesh.nCells: {mesh.numberOfCells}")
    print(f"[DEBUG] r_inner: {r_inner:.4f}, r_outer: {r_outer:.4f}")

    T = fp.CellVariable(mesh=mesh, name="temperature", value=T_inf)
    print(f"[DEBUG] T.shape: {T.shape}, T.value[:5]: {T.value[:5]}")

    T.faceGrad.constrain((-q_inner / k,), where=mesh.facesLeft)
    print(f"[DEBUG] Imposed Neumann at mesh.facesLeft (q_inner/k={q_inner/k:.2f})")

    robin_coeff = fp.CellVariable(mesh=mesh, value=0.0)
    robin_coeff[-1] = h / k
    print(f"[DEBUG] robin_coeff shape: {robin_coeff.shape}, value[-5:]: {robin_coeff.value[-5:]}")

    eq = (
        fp.DiffusionTerm(coeff=k)
        + fp.ImplicitSourceTerm(coeff=robin_coeff)
    )

    print("[DEBUG] Ready to solve...")
    eq.solve(var=T)
    print("[DEBUG] Solution complete.")
    print(f"[DEBUG] T.value[:10]: {T.value[:10]}")
    print(f"[DEBUG] T min: {np.min(T.value)}, max: {np.max(T.value)}")
    return T

