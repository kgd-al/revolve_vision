import logging
import os
from typing import Tuple

import mujoco
import numpy as np
from mujoco import MjModel, MjData

from ..misc.config import Config

logger = logging.getLogger(__name__)


# ==============================================================================
# Vision (through offscreen OpenGL Rendering)
# ==============================================================================

class OpenGLVision:
    max_width, max_height = 2000, 2000  # Size of the back-buffer (not used with egl)
    global_context = None

    def __init__(self, model: MjModel, shape: Tuple[int, int], headless: bool = True):
        # if OpenGLVision.global_context is None:
        if OpenGLVision.global_context is None and headless:
            OGL = Config.OpenGLLib
            ogl = Config.opengl_lib.upper()
            if ogl == OGL.GLFW.name: # Does not work in multithread
                from mujoco.glfw import GLContext
            elif ogl == OGL.EGL.name:
                from mujoco.egl import GLContext
                os.environ['MUJOCO_GL'] = 'egl'
            elif ogl == OGL.OSMESA.name:
                from mujoco.osmesa import GLContext
                os.environ['MUJOCO_GL'] = 'osmesa'
            else:
                raise ValueError(f"Unknown OpenGL backend {Config.opengl_lib}")
            OpenGLVision.global_context = GLContext(self.max_width, self.max_height)
            OpenGLVision.global_context.make_current()
            logger.info(f"Initialized {OpenGLVision.global_context=}")

        w, h = shape
        assert 0 < w <= self.max_width
        assert 0 < h <= self.max_height

        logger.info(f"New {self}")

        self.width, self.height = w, h
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.viewport = mujoco.MjrRect(0, 0, w, h)

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 1

        self.vopt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=model.ngeom+2)
        self.pert = mujoco.MjvPerturb()

        self.img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def process(self, model, data):
        """
        Grab pixels from the underlying opengl context

         First updates the mujoco scene using the `model` and `data`, then
         renders it to an offscreen buffer and grab its contents

         :param model: the mujoco static data
         :param data: the mujoco dynamical data

         :return: a numpy array (RGB format)
        """
        # mujoco.mjv_updateScene(
        #     model, data,
        #     self.vopt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL.value,
        #     self.scene)
        # mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        # mujoco.mjr_render(self.viewport, self.scene, self.context)
        # mujoco.mjr_readPixels(self.img, None, self.viewport, self.context)

        return self.img

    def __del__(self):
        self.context.free()
        logger.info(f"Del {self}")
