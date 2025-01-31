import grid
import critter
import nengo
import nengo.spa as spa
import matplotlib.pyplot as plt 
import numpy as np

save_figs = False
fig_path = "plots/"

# ---- plotting functions ----
def recalled_sequence(sim, save_figs=False):
    # Plot recalled sequence
    plt.figure()
    plt.plot(sim.trange(), sim.data[sequence_probe])
    plt.xlabel("Time (s)")
    plt.ylabel("Recalled Sequence Activity")
    plt.title("Sequence Recall Over Time")
    plt.show()
    if save_figs:
        plt.savefig(fig_path + "sequence_recall.png")

# ---- Test Functions ----

def sequence_recall(sim):
    pass

#  Overview of probes in model
#  - sequence_probe: Probes the output of the sequence memory


if __name__ == "__main__":
    # Create the critter
    critter = critter.Critter()

    # Create the simulator
    sim = nengo.Simulator(critter.model)

    # Run the simulation
    with sim:
        sim.run(1.0)

    # Plot the world
    plt.imshow(critter.world.draw(), interpolation='nearest')
    plt.show()

    # Plot the recalled sequence
    recalled_sequence(sim)

