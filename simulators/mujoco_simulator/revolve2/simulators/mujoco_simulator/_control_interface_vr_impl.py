from revolve2.simulation.scene import ControlInterface, JointHinge, UUIDKey
from revolve2.vr.server import Revolve2Server
from ._abstraction_to_mujoco_mapping import AbstractionToMujocoMapping, JointHingeMujoco


class ControlInterfaceVRImpl(ControlInterface):
    """Implementation of the control interface for MuJoCo."""

    _connection: Revolve2Server
    _abstraction_to_mujoco_mapping: AbstractionToMujocoMapping

    def __init__(
            self,
            connection: Revolve2Server,
            abstraction_to_mujoco_mapping: AbstractionToMujocoMapping,
    ) -> None:
        """
        Initialize this object.

        :param connection:
        :param abstraction_to_mujoco_mapping: A mapping between simulation abstraction and mujoco.
        """
        self._connection = connection
        self._abstraction_to_mujoco_mapping = abstraction_to_mujoco_mapping

    def set_joint_hinge_position_target(
            self, joint_hinge: JointHinge, position: float
    ) -> None:
        """
        Send the position target of a hinge joint.

        :param joint_hinge: The hinge to set the position target for.
        :param position: The position target.
        """
        maybe_hinge_joint_mujoco = self._abstraction_to_mujoco_mapping.hinge_joint.get(
            UUIDKey(joint_hinge)
        )
        assert (
                maybe_hinge_joint_mujoco is not None
        ), "Hinge joint does not exist in this scene."
        self._connection.append_control_command(maybe_hinge_joint_mujoco.ctrl_index_position, position,
                                                maybe_hinge_joint_mujoco.ctrl_index_velocity, 0.0)

    def get_hinge_joint_mujoco(self, joint_hinge: JointHinge) -> JointHingeMujoco:
        maybe_hinge_joint_mujoco = self._abstraction_to_mujoco_mapping.hinge_joint.get(
            UUIDKey(joint_hinge)
        )
        return maybe_hinge_joint_mujoco
