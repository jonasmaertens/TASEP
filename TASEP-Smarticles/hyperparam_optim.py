from Trainer import Trainer, Hyperparams, EnvParams
import itertools
from tqdm import tqdm
import datetime

observation_distances = [1, 2]
batch_sizes = [512, 8192]
gammas = [0.3, 0.6, 0.999]
eps_starts = [0.9]
eps_ends = [0.1]
eps_decays = [5_000, 100_000, 600_000]
taus = [1e-4, 1e-2]
lrs = [1e-3, 1e-2, 1e-1]
memory_sizes = [1000, 1_000_000]

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    with open(f"data/hyperparam_tune/models{timestamp}.csv", "w") as f:
        f.write("observation_distance,batch_size,gamma,eps_start,eps_end,eps_decay,tau,lr,memory_size,current\n")
    for observation_distance, batch_size, gamma, eps_start, eps_end, eps_decay, tau, lr, memory_size in tqdm(
            itertools.product(
                    observation_distances, batch_sizes, gammas, eps_starts, eps_ends, eps_decays, taus, lrs, memory_sizes)):
        envParams = EnvParams(render_mode=None,
                              length=32,
                              width=8,
                              moves_per_timestep=20,
                              window_height=256,
                              observation_distance=observation_distance,
                              initial_state_template="checkerboard",
                              distinguishable_particles=False)
        hyperparams = Hyperparams(BATCH_SIZE=batch_size,
                                  GAMMA=gamma,
                                  EPS_START=eps_start,
                                  EPS_END=eps_end,
                                  EPS_DECAY=eps_decay,
                                  TAU=tau,
                                  LR=lr,
                                  MEMORY_SIZE=memory_size)

        trainer = Trainer(envParams, hyperparams, reset_interval=3000,
                          total_steps=30000, plot_interval=1000, do_plot=False)

        trainer.train()

        trainer.save(
            f"models/hyperparam_tune/example_model_{observation_distance}_{batch_size}_{gamma}_{eps_start}_{eps_end}_{eps_decay}_{tau}_{lr}_{memory_size}.pt")

        trainer.save_plot(
            f"plots/hyperparam_tune/example_model_{observation_distance}_{batch_size}_{gamma}_{eps_start}_{eps_end}_{eps_decay}_{tau}_{lr}_{memory_size}.png",
            append_timestamp=False)

        with open(f"data/hyperparam_tune/models{timestamp}.csv", "a") as f:
            f.write(f"{observation_distance},{batch_size},{gamma},{eps_start},{eps_end},{eps_decay},{tau},{lr},{memory_size},{trainer.currents[-1]}\n")
