import torch
from pytorch_mppi import MPPI

from pytorch_mppi import autotune

def evaluate():
    costs = []
    rollouts = []
    # we sample multiple trajectories for the same start to goal problem, but in your case you should consider
    # evaluating over a diverse dataset of trajectories
    for j in range(num_trajectories):
        mppi.U = nominal_trajectory.clone()
        # the nominal trajectory at the start will be different if the horizon's changed
        mppi.change_horizon(mppi.T)
        # usually MPPI will have its nominal trajectory warm-started from the previous iteration
        # for a fair test of tuning we will reset its nominal trajectory to the same random one each time
        # we manually warm it by refining it for some steps
        for k in range(num_refinement_steps):
            mppi.command(env.start, shift_nominal_trajectory=False)

        rollout = mppi.get_rollouts(env.start)

        this_cost = 0
        rollout = rollout[0]
        # here we evaluate on the rollout MPPI cost of the resulting trajectories
        # alternative costs for tuning the parameters are possible, such as just considering terminal cost
        if evaluate_running_cost:
            for t in range(len(rollout) - 1):
                this_cost = this_cost + env.running_cost(rollout[t], mppi.U[t])
        this_cost = this_cost + env.terminal_cost(rollout, mppi.U)

        rollouts.append(rollout)
        costs.append(this_cost)
    # can return None for rollouts if they do not need to be calculated
    return autotune.EvaluationResult(torch.stack(costs), torch.stack(rollouts))

if __name__ == '__main__':
    device = "cpu"
    dtype = torch.double

    # create toy environment to do on control on (default start and goal)
    env = Toy2DEnvironment(visualize=True, terminal_scale=10)

    # create MPPI with some initial parameters
    mppi = MPPI(env.dynamics, env.running_cost, 2,
            terminal_state_cost=env.terminal_cost,
            noise_sigma=torch.diag(torch.tensor([5., 5.], dtype=dtype, device=device)),
            num_samples=500,
            horizon=20, device=device,
            u_max=torch.tensor([2., 2.], dtype=dtype, device=device),
            lambda_=1)
    
    # use the same nominal trajectory to start with for all the evaluations for fairness
    nominal_trajectory = mppi.U.clone()
    # parameters for our sample evaluation function - lots of choices for the evaluation function
    evaluate_running_cost = True
    num_refinement_steps = 10
    num_trajectories = 5