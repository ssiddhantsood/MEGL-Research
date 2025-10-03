# demo_rgn.py
import os
import numpy as np
from utils import (
    random_geometric_graph_2d,
    random_omega,
    create_output_folder,
    draw_geometric_network_graph,
    continuation_descend_K,
    plot_r_vs_K,
)

def demo_random_geometric_network():
    # make output folder
    outdir = create_output_folder("runs_demo")

    # generate random geometric network with positions
    n = 30
    W, pos = random_geometric_graph_2d(n=n, radius=20, return_pos=True)

    # assign random natural frequencies
    omega = random_omega(n, std=1.0)

    # plot the geometric network
    net_path = os.path.join(outdir, "rgn_network.png")
    draw_geometric_network_graph(W, omega, pos, net_path, title=f"RGN with n={n}")

    # run continuation to get synchronization curve
    Kc, Ks, Rs = continuation_descend_K(omega, W, K_start=10.0, K_min=0.02, K_step=0.1)

    # plot r(K)
    r_path = os.path.join(outdir, "rgn_r_vs_K.png")
    plot_r_vs_K(Ks, Rs, r_path, Kc=Kc, title="RGN Synchronization Curve")

    print(f"✅ Demo complete. Plots saved in: {outdir}")
    print(f"   - Network: {net_path}")
    print(f"   - r(K):    {r_path}")
    if Kc is not None:
        print(f"   Estimated Kc ≈ {Kc:.3f}")
    else:
        print("   Could not estimate Kc")

if __name__ == "__main__":
    demo_random_geometric_network()
