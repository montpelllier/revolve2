"""Evaluator class."""

import math

import numpy as np
import numpy.typing as npt
from pyrr import Vector3

from revolve2.standards import fitness_functions, terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, CpgNetworkStructure
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulation.scene import Pose
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.vr.server import Revolve2Server


class Evaluator:
    """Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain
    _cpg_network_structure: CpgNetworkStructure
    _body: Body
    _output_mapping: list[tuple[int, ActiveHinge]]
    _connection: Revolve2Server

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
        cpg_network_structure: CpgNetworkStructure,
        body: Body,
        output_mapping: list[tuple[int, ActiveHinge]],
        connection: Revolve2Server | None = None,
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        :param cpg_network_structure: Cpg structure for the brain.
        :param body: Modular body of the robot.
        :param output_mapping: A mapping between active hinges and the index of their corresponding cpg in the cpg network structure.
        """
        self._simulator = LocalSimulator(
            viewer_type="native"
        )
        self._terrain = terrains.flat()
        self._cpg_network_structure = cpg_network_structure
        self._body = body
        self._output_mapping = output_mapping
        self._connection = connection

    def evaluate(
        self,
        solutions: list[npt.NDArray[np.float_]],
        onSimulateFinish = None
    ) -> npt.NDArray[np.float_]:
        """
        Evaluate multiple robots.

        Fitness is the distance traveled on the xy plane.

        :param solutions: Solutions to evaluate.
        :returns: Fitnesses of the solutions.
        """
        # Create robots from the brain parameters.
        robots = [
            ModularRobot(
                body=self._body,
                brain=BrainCpgNetworkStatic.uniform_from_params(
                    params=params,
                    cpg_network_structure=self._cpg_network_structure,
                    initial_state_uniform=math.sqrt(2) * 0.5,
                    output_mapping=self._output_mapping,
                ),
            )
            for params in solutions
        ]

        # Create the scenes.
        # scenes = []
        scene = ModularRobotScene(terrain=self._terrain)
        poses = [
            Pose(Vector3([2.0, 1.0, 0.0])),
            Pose(Vector3([1.0, 1.0, 0.0])),
            Pose(Vector3([0.0, 1.0, 0.0])),
            Pose(Vector3([-1.0, 1.0, 0.0])),
            Pose(Vector3([-2.0, 1.0, 0.0])),
            Pose(Vector3([2.0, -1.0, 0.0])),
            Pose(Vector3([1.0, -1.0, 0.0])),
            Pose(Vector3([0.0, -1.0, 0.0])),
            Pose(Vector3([-1.0, -1.0, 0.0])),
            Pose(Vector3([-2.0, -1.0, 0.0])),
        ]
        for index, robot in enumerate(robots):

            scene.add_robot(robot, pose=poses[index])

        def onResult(scene_states):
            # Calculate the xy displacements.
            xy_displacements = [
                fitness_functions.xy_displacement(
                    scene_states[0].get_modular_robot_simulation_state(robot),
                    scene_states[-1].get_modular_robot_simulation_state(robot),
                )
                for robot in robots
            ]

            if onSimulateFinish is not None:
                onSimulateFinish(np.array(xy_displacements))


        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scene,
            vr=True,
            connection = self._connection,
            onResultHandler = onResult
        )

        return np.array([])
