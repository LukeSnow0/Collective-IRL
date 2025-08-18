# radar_collective_rp.py: 

This script simulates assignable per-agent choices under a common probe/price vector and reconstructs each agent’s utility via Afriat's Theorem. 
Because each agent’s bundle is observed at every time step (probe–response pairs), agents are identified without label ambiguity. 

## What this project does

1. **Simulate data (generator):**  
   For `t = 1..N`, draw a probe vector `a_t` and solve a constrained nonlinear program that allocates resources across `M` agents, maximizing a weighted sum of known utility shapes (used only for simulation).
    Constraint:  
   \[
   \sum_{j=1}^M a_t^\top\, b_{j,t} \le y_t.
   \]

3. **Reconstruct utilities (Afriat per agent):**  
   For each agent \( j \), solve the Afriat LP to obtain \( \{u_{j,t}, \lambda_{j,t}\}_{t=1}^N \) such that:
   \[
   u_{j,t} \le u_{j,s} + \lambda_{j,s}\, a_s^\top\big(b_{j,t} - b_{j,s}\big) - \phi_j \quad \forall s,t.
   \]
   This yields a piecewise-linear concave utility representation for that agent.

4. **Diagnostics & plots:**  
   Prints the max Afriat violation per agent (should be ~≤ 0 up to tolerance) and plots **Agent j Reconstructed Utility** vs **Agent j True Utility**.

---

## Quickstart

```bash
# 1) Create a fresh environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -U numpy scipy matplotlib

# 3) Run the script
python radar_collective_rp.py
