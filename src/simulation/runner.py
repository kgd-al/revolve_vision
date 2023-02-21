import math
import os.path
import sys
from abc import ABC
from ast import literal_eval
from collections import namedtuple
from configparser import ConfigParser
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import glfw
import mujoco
from mujoco import MjModel, MjData
from mujoco_viewer import mujoco_viewer
from pyrr import Vector3, Quaternion
from revolve2.core.modular_robot import ModularRobot
from revolve2.core.physics.environment_actor_controller import \
    EnvironmentActorController
from revolve2.core.physics.running import Environment
from revolve2.core.physics.running import PosedActor, ActorControl
from revolve2.runners.mujoco import LocalRunner
from revolve2.runners.mujoco._local_runner import mjcf

from ..misc.config import Config


class CallbackType(Enum):
    POST_CONTROL_STEP = 0


RunnerCallback = Callable[[float, MjModel, MjData], None]
RunnerCallbacks = Dict[CallbackType, RunnerCallback]


@dataclass
class RunnerOptions:
    @dataclass
    class View:
        start_paused: bool = False
        speed: float = 1.0
        auto_quit: bool = True
        cam_id: Optional[int] = 0
        settings_restore: bool = True
        settings_save: bool = True
        mark_start: bool = True

    view: Optional[View] = None

    @dataclass
    class Record:
        video_file_path: Path

        width: int = 640
        height: int = 480

        fps: int = 24
    record: Optional[Record] = None


class DefaultActorControl(ActorControl):
    def __init__(self, mj_model: MjModel, mj_data: MjData):
        ActorControl.__init__(self)


class DefaultEnvironmentActorController(EnvironmentActorController):
    def control(self, dt: float, actor_control: DefaultActorControl) -> None:
        self.actor_controller.step(dt, actor_control)
        actor_control.set_dof_targets(0, self.actor_controller.get_dof_targets())


class Runner(LocalRunner):
    environmentActorController_t = DefaultEnvironmentActorController
    actorController_t = DefaultActorControl

    def __init__(self, robot: ModularRobot, options: RunnerOptions,
                 env_seeder: Callable[[mjcf.RootElement, RunnerOptions], None],
                 callbacks: Optional[RunnerCallbacks] = None):

        LocalRunner.__init__(self)

        self.options = options

        self.callbacks = {}
        if callbacks is not None:
            self.callbacks.update(callbacks)

        actor, controller = robot.make_actor_and_controller()
        bounding_box = actor.calc_aabb()
        env = Environment(self.environmentActorController_t(controller))
        env.actors.append(
            PosedActor(
                actor,
                Vector3(
                    [
                        0.0,
                        0.0,
                        bounding_box.size.z / 2.0 - bounding_box.offset.z,
                    ]
                ),
                Quaternion(),
                [0.0 for _ in controller.get_dof_targets()],
            )
        )
        self.controller = env.controller

        self.model = mujoco.MjModel.from_xml_string(
            LocalRunner._make_mjcf(env,
                                   partial(env_seeder, options=options)))

        # TODO initial dof state
        self.data = mujoco.MjData(self.model)

        initial_targets = [
            dof_state
            for posed_actor in env.actors
            for dof_state in posed_actor.dof_states
        ]
        LocalRunner._set_dof_targets(self.data, initial_targets)

        self.viewer = None
        self.video = None
        self.headless = options is None or (options.view is None and options.record is None)
        if not self.headless:
            self.prepare_view(options)

    def prepare_view(self, options):
        record = (options.record is not None)
        width = height = None
        if record:
            width, height = options.record.width, options.record.height

        viewer = mujoco_viewer.MujocoViewer(
            self.model,
            self.data,
            'offscreen' if record else 'window',
            width=width, height=height,
        )
        viewer._render_every_frame = False  # Private but functionality is not exposed and for now it breaks nothing.
        viewer._paused = False if record else options.view.start_paused

        if record:
            self.video = namedtuple('Video', ['step', 'writer', 'last'])
            self.video.step = 1 / options.record.fps
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video.writer = cv2.VideoWriter(
                str(options.record.video_file_path),
                fourcc,
                options.record.fps,
                (viewer.viewport.width, viewer.viewport.height),
            )

            viewer._hide_menu = True

            self.video.last = 0.0
        else:
            viewer._run_speed = options.view.speed

        cam_id = -1
        if options.view is not None and options.view.cam_id is not None:
            cam_id = options.view.cam_id
        elif record:
            cam_id = 0
        if cam_id != -1:
            viewer.cam.fixedcamid = cam_id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        self.viewer = viewer

        if self.options.view is not None:
            if self.options.view.settings_restore:
                self.restore_settings()
            if self.options.view.settings_save:
                glfw.set_window_close_callback(viewer.window,
                                               lambda _: self.save_settings())

    @staticmethod
    def _config_file():
        return os.path.join(os.path.expanduser('~'),
                            '.config/revolve/viewer.ini')

    def restore_settings(self):
        config = ConfigParser()
        config.read(self._config_file())
        if 'values' in config:
            values = config['values']
            glfw.restore_window(self.viewer.window)
            if 'size' in values:
                glfw.set_window_size(self.viewer.window,
                                     *literal_eval(values['size']))
            if 'pos' in values:
                glfw.set_window_pos(self.viewer.window,
                                    *literal_eval(values['pos']))
            if 'menu' in values:
                self.viewer._hide_menus = (values['menu'] == "False")

    def save_settings(self):
        config = ConfigParser()
        config['values'] = {}
        values = config['values']
        values['pos'] = str(glfw.get_window_pos(self.viewer.window))
        values['size'] = str(glfw.get_window_size(self.viewer.window))
        values['menu'] = str(not self.viewer._hide_menus)
        path = Path(self._config_file())
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            config.write(f)

    def update_view(self, time):
        if self.video is None:
            self.viewer.render()

        # capture video frame if it's time
        if self.video is not None and time >= self.video.last + self.video.step:
            self.video.last = int(time / self.video.step) * self.video.step

            img = self.viewer.read_pixels()
            self.video.writer.write(img)

    def close_view(self):
        if self.options.view is not None and self.options.view.auto_quit:
            self.save_settings()
            self.viewer.close()

        if self.options.record is not None:
            self.video.writer.release()

    def run(self) -> None:
        last_control_time = 0.0

        control_step = 1 / Config.control_frequency

        while (time := self.data.time) < Config.simulation_time:
            # do control if it is time
            is_control_step = False
            if time >= last_control_time + control_step:
                is_control_step = True
                last_control_time = \
                    math.floor(time / control_step) * control_step
                control_user = self.actorController_t(self.model, self.data)
                self.controller.control(control_step, control_user)
                actor_targets = control_user._dof_targets
                actor_targets.sort(key=lambda t: t[0])
                targets = [
                    target
                    for actor_target in actor_targets
                    for target in actor_target[1]
                ]
                LocalRunner._set_dof_targets(self.data, targets)

            # step simulation
            mujoco.mj_step(self.model, self.data)

            if is_control_step and \
                    CallbackType.POST_CONTROL_STEP in self.callbacks:
                self.callbacks[CallbackType.POST_CONTROL_STEP](control_step, self.model, self.data)

            if not self.headless:
                self.update_view(time)

        if not self.headless:
            self.close_view()

    def get_actor_state(self, robot_index):
        return self._get_actor_state(robot_index, self.data, self.model)