import logging
import math

import cv2
import mujoco
import numpy as np
import numpy.typing as npt

from revolve2.modular_robot.brain.cpg import BrainCpgInstance
from revolve2.modular_robot_simulation._build_multi_body_systems import BodyToMultiBodySystemMapping
from revolve2.simulation.scene import Scene, SimulationState, ControlInterface, UUIDKey
from revolve2.simulation.simulator import RecordSettings
from revolve2.vr.server import Revolve2Server
from ._control_interface_impl import ControlInterfaceImpl
from ._control_interface_vr_impl import ControlInterfaceVRImpl
from ._open_gl_vision import OpenGLVision
from ._render_backend import RenderBackend
from ._scene_to_model import scene_to_xml
from ._simulation_state_impl import SimulationStateImpl
from .viewers import CustomMujocoViewer, NativeMujocoViewer, ViewerType


def simulate_scene_vr_v2(
        scene_id: int,
        scene: Scene,
        headless: bool,
        record_settings: RecordSettings | None,
        start_paused: bool,
        control_step: float,
        sample_step: float | None,
        simulation_time: int | None,
        simulation_timestep: float,
        cast_shadows: bool,
        fast_sim: bool,
        viewer_type: ViewerType,
        render_backend: RenderBackend = RenderBackend.EGL,
        connection: Revolve2Server | None = None,
) -> list[SimulationState]:
    """
    Simulate a scene.

    :param scene_id: An id for this scene, unique between all scenes ran in parallel.
    :param scene: The scene to simulate.
    :param headless: If False, a viewer will be opened that allows a user to manually view and manually interact with the simulation.
    :param record_settings: If not None, recording will be done according to these settings.
    :param start_paused: If true, the simulation will start in a paused state. Only makessense when headless is False.
    :param control_step: The time between each call to the handle function of the scene handler. In seconds.
    :param sample_step: The time between each state sample of the simulation. In seconds.
    :param simulation_time: How long to simulate for. In seconds.
    :param simulation_timestep: The duration to integrate over during each step of the simulation. In seconds.
    :param cast_shadows: If shadows are cast.
    :param fast_sim: If fancy rendering is disabled.
    :param viewer_type: The type of viewer used for the rendering in a window.
    :param render_backend: The backend to be used for rendering (EGL by default and switches to GLFW if no cameras are on the robot).
    :param connection: If not None, a connection to the simulator will be created.
    :returns: The results of simulation. The number of returned states depends on `sample_step`.
    :raises ValueError: If the viewer is not able to record.
    """
    logging.info(f"Simulating scene {scene_id}")

    """Define mujoco data and model objects for simulating."""
    model, mapping, xml = scene_to_xml(
        scene, simulation_timestep, cast_shadows=cast_shadows, fast_sim=fast_sim
    )

    if xml and connection:
        def handle_message(message):
            if not message:
                return
            print(f'from handle_message: {message}')
            msg_type = message.get("type")
            msg_data = message.get("data")

            if msg_type == "start_simulation":
                data = mujoco.MjData(model)

                control_interface = ControlInterfaceVRImpl(
                    connection=connection, abstraction_to_mujoco_mapping=mapping
                )
                """Define some additional control variables."""
                last_control_time = 0.0
                control_states: list[SimulationState] = (
                    []
                )
                camera_viewers = {
                    camera.camera_id: OpenGLVision(
                        model=model, camera=camera, headless=headless, open_gl_lib=render_backend
                    )
                    for camera in mapping.camera_sensor.values()
                }
                mujoco.mj_forward(model, data)
                images = {
                    camera_id: camera_viewer.process(model, data)
                    for camera_id, camera_viewer in camera_viewers.items()
                }
                while (time := data.time) < (
                        float("inf") if simulation_time is None else simulation_time
                ):
                    mujoco.mj_step(model, data)
                    # do control if it is time
                    if time >= last_control_time + control_step:
                        last_control_time = math.floor(time / control_step) * control_step

                        simulation_state = SimulationStateImpl(
                            data=data, abstraction_to_mujoco_mapping=mapping, camera_views=images
                        )
                        scene.handler.handle(simulation_state, control_interface, control_step)
                        control_states.append(simulation_state)
                        connection.send_control_commands()

            elif msg_type == "send_brains":
                data = mujoco.MjData(model)

                control_interface = ControlInterfaceVRImpl(
                    connection=connection, abstraction_to_mujoco_mapping=mapping
                )

                scene_brains = scene.handler.get_brains()
                for brain_instance, body_to_multi_body_system_mapping in scene_brains:
                    print('converting brain to vr brain')
                    brain_data = create_with_brain_instance(brain_instance, control_interface,
                                                            body_to_multi_body_system_mapping)

                    connection.append_brain_data(brain_data)

                # x = [[0.1, 0.2,],[0.1,0.0]]
                # data.xpos = x

                # return
                connection.send_brains()

        connection.handler = handle_message
        connection.send_mujoco_xml(xml)
        return []

    data = mujoco.MjData(model)

    """Define a control interface for the mujoco simulation (used to control robots)."""
    control_interface = ControlInterfaceImpl(
        data=data, abstraction_to_mujoco_mapping=mapping
    )
    """Make separate viewer for camera sensors."""
    camera_viewers = {
        camera.camera_id: OpenGLVision(
            model=model, camera=camera, headless=headless, open_gl_lib=render_backend
        )
        for camera in mapping.camera_sensor.values()
    }

    """Define some additional control variables."""
    last_control_time = 0.0
    last_sample_time = 0.0
    last_video_time = 0.0  # time at which last video frame was saved

    simulation_states: list[SimulationState] = (
        []
    )  # The measured states of the simulation

    """If we dont have cameras and the backend is not set we go to the default GLFW."""
    if len(mapping.camera_sensor.values()) == 0:
        render_backend = RenderBackend.GLFW

    """Initialize viewer object if we need to render the scene."""
    if not headless or record_settings is not None:
        match viewer_type:
            case viewer_type.CUSTOM:
                viewer = CustomMujocoViewer
            case viewer_type.NATIVE:
                viewer = NativeMujocoViewer
            case _:
                raise ValueError(
                    f"Viewer of type {viewer_type} not defined in _simulate_scene."
                )

        viewer = viewer(
            model,
            data,
            width=None if record_settings is None else record_settings.width,
            height=None if record_settings is None else record_settings.height,
            backend=render_backend,
            start_paused=start_paused,
            render_every_frame=False,
            hide_menus=(record_settings is not None),
        )

    """Record the scene if we want to record."""
    if record_settings is not None:
        if not viewer.can_record:
            raise ValueError(
                f"Selected Viewer {type(viewer).__name__} has no functionality to record."
            )
        video_step = 1 / record_settings.fps
        video_file_path = f"{record_settings.video_directory}/{scene_id}.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video = cv2.VideoWriter(
            video_file_path,
            fourcc,
            record_settings.fps,
            viewer.current_viewport_size(),
        )

    """
    Compute forward dynamics without actually stepping forward in time.
    This updates the data so we can read out the initial state.
    """

    mujoco.mj_forward(model, data)
    images = {
        camera_id: camera_viewer.process(model, data)
        for camera_id, camera_viewer in camera_viewers.items()
    }

    # Sample initial state.
    if sample_step is not None:
        simulation_states.append(
            SimulationStateImpl(
                data=data, abstraction_to_mujoco_mapping=mapping, camera_views=images
            )
        )

    """After rendering the initial state, we enter the rendering loop."""
    while (time := data.time) < (
            float("inf") if simulation_time is None else simulation_time
    ):

        # do control if it is time
        if time >= last_control_time + control_step:
            last_control_time = math.floor(time / control_step) * control_step

            simulation_state = SimulationStateImpl(
                data=data, abstraction_to_mujoco_mapping=mapping, camera_views=images
            )
            scene.handler.handle(simulation_state, control_interface, control_step)

        # sample state if it is time
        if sample_step is not None:
            if time >= last_sample_time + sample_step:
                last_sample_time = int(time / sample_step) * sample_step
                simulation_states.append(
                    SimulationStateImpl(
                        data=data,
                        abstraction_to_mujoco_mapping=mapping,
                        camera_views=images,
                    )
                )

        # step simulation
        mujoco.mj_step(model, data)
        # extract images from camera sensors.
        images = {
            camera_id: camera_viewer.process(model, data)
            for camera_id, camera_viewer in camera_viewers.items()
        }

        # render if not headless. also render when recording and if it time for a new video frame.
        if not headless or (
                record_settings is not None and time >= last_video_time + video_step
        ):
            _status = viewer.render()

            # Check if simulation was closed
            if _status == -1:
                break

        # capture video frame if it's time
        if record_settings is not None and time >= last_video_time + video_step:
            last_video_time = int(time / video_step) * video_step

            # https://github.com/deepmind/mujoco/issues/285 (see also record.cc)
            img: npt.NDArray[np.uint8] = np.empty(
                (*viewer.current_viewport_size(), 3),
                dtype=np.uint8,
            )

            mujoco.mjr_readPixels(
                rgb=img,
                depth=None,
                viewport=viewer.view_port,
                con=viewer.context,
            )
            # Flip the image and map to OpenCV colormap (BGR -> RGB)
            img = np.flipud(img)[:, :, ::-1]
            video.write(img)

    """Once simulation is done we close the potential viewer and release the potential video."""
    if not headless or record_settings is not None:
        viewer.close_viewer()

    if record_settings is not None:
        video.release()

    # Sample one final time.
    if sample_step is not None:
        simulation_states.append(
            SimulationStateImpl(
                data=data, abstraction_to_mujoco_mapping=mapping, camera_views=images
            )
        )

    logging.info(f"Scene {scene_id} done.")
    return simulation_states


def create_with_brain_instance(
        brain_cpg_instance: BrainCpgInstance,
        control_interface: ControlInterface,
        body_to_multi_body_system_mapping: BodyToMultiBodySystemMapping
):
    output_mapping = []

    state, weight_matrix, tmp_mapping = brain_cpg_instance.data_copy()
    for state_index, active_hinge in tmp_mapping:
        joint_hinge = body_to_multi_body_system_mapping.active_hinge_to_joint_hinge[UUIDKey(active_hinge)]
        hinge_joint_mujoco = control_interface.get_hinge_joint_mujoco(joint_hinge)

        output_mapping.append(
            (state_index, active_hinge.range, hinge_joint_mujoco)
        )
    brain_data = data_to_dict(state, weight_matrix, output_mapping)
    print(brain_data)
    # brain_vr_instance = BrainCpgVr(initial_state=state, weight_matrix=weight_matrix, output_mapping=output_mapping)

    return brain_data


def data_to_dict(initial_state, weight_matrix, output_mapping):
    # Convert output_mapping into a serializable format
    output_mapping_serializable = [
        {
            "state_index": state_index,
            "hinge_range": hinge_range,
            "joint_hinge_mujoco": {
                "id": hinge.id,
                "ctrl_index_position": hinge.ctrl_index_position,
                "ctrl_index_velocity": hinge.ctrl_index_velocity
            }
        }
        for state_index, hinge_range, hinge in output_mapping
    ]

    # Create a dictionary representation
    brain_dict = {
        "initial_state": initial_state.tolist(),  # Convert NumPy array to list
        "weight_matrix": weight_matrix.tolist(),  # Convert NumPy matrix to list
        "output_mapping": output_mapping_serializable  # Use serializable version
    }

    # Convert dictionary to JSON
    return brain_dict
