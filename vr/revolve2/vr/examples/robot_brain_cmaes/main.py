"""Main script for the example."""

import logging

import cma
import revolve2.vr.examples.robot_brain_cmaes.config as config
# import config
from revolve2.vr.examples.robot_brain_cmaes.evaluator import Evaluator

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import seed_from_time
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)
from revolve2.vr.server import Revolve2Server


def main(connection: Revolve2Server | None = None) -> None:
    """Run the experiment."""
    setup_logging(file_name="log.txt")

    # Find all active hinges in the body
    active_hinges = config.BODY.find_modules_of_type(ActiveHinge)

    # Create a structure for the CPG network from these hinges.
    # This also returns a mapping between active hinges and the index of there corresponding cpg in the network.
    (
        cpg_network_structure,
        output_mapping,
    ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

    # Intialize the evaluator that will be used to evaluate robots.
    evaluator = Evaluator(
        headless=True,
        num_simulators=config.NUM_SIMULATORS,
        cpg_network_structure=cpg_network_structure,
        body=config.BODY,
        output_mapping=output_mapping,
        connection=connection
    )

    # Initial parameter values for the brain.
    initial_mean = cpg_network_structure.num_connections * [0.5]

    # We use the CMA-ES optimizer from the cma python package.
    # Initialize the cma optimizer.
    options = cma.CMAOptions()
    options.set("bounds", [-1.0, 1.0])
    # The cma package uses its own internal rng.
    # Instead of creating our own numpy rng, we use our seed to initialize cma.
    rng_seed = seed_from_time() % 2**32  # Cma seed must be smaller than 2**32.
    options.set("seed", rng_seed)
    opt = cma.CMAEvolutionStrategy(initial_mean, config.INITIAL_STD, options)

    generation_index = 0

    # Run cma for the defined number of generations.
    logging.info("Start optimization process.")
    print("Start optimization process.")
    logging.info(f"Generation {generation_index + 1} / {config.NUM_GENERATIONS}.")
    print(f"Generation {generation_index + 1} / {config.NUM_GENERATIONS}.")
    solutions = opt.ask()
    print(f"solutions: {solutions}")
    # length = min(4, len(solutions))
    # solutions = solutions[:length]

    def onSimulateResult(result):
        nonlocal generation_index
        nonlocal solutions
        fitnesses = -result
        # Tell cma the fitnesses.
        opt.tell(solutions, fitnesses)

        logging.info(f"{opt.result.xbest=} {opt.result.fbest=}")
        print(f"{opt.result.xbest=} {opt.result.fbest=}")

        if generation_index < config.NUM_GENERATIONS:
            logging.info(f"Generation {generation_index + 1} / {config.NUM_GENERATIONS}.")
            print(f"Generation {generation_index + 1} / {config.NUM_GENERATIONS}.")

            # Increase the generation index counter.
            generation_index += 1
            solutions = opt.ask()
            print(f"solutions: {solutions}")

            # Get the sampled solutions(parameters) from cma.
            evaluator.evaluate(solutions, onSimulateResult)


    evaluator.evaluate(solutions, onSimulateResult)


if __name__ == "__main__":
    main()
