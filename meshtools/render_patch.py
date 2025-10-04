from OpenGL.GL import (
    glBindFramebuffer,
    glReadBuffer,
    GL_READ_FRAMEBUFFER,
    GL_FRONT,
    glReadPixels,
    GL_DEPTH_COMPONENT,
    GL_FLOAT,
    GL_DRAW_FRAMEBUFFER,
    glBlitFramebuffer,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_LINEAR,
    GL_NEAREST,
    GL_RGBA,
    GL_RGB,
    GL_UNSIGNED_BYTE,
)
import sys
import numpy as np
from pyrender import Renderer, OffscreenRenderer
from pyrender.constants import RenderFlags


class MyRender(Renderer):
    def read_depth_buf(self):
        """Read and return the current viewport's color buffer.

        Returns
        -------
        depth_im : (h, w) float32
            The depth buffer in linear units.
        """
        width, height = self.viewport_width, self.viewport_height
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        glReadBuffer(GL_FRONT)
        depth_buf = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)

        depth_im = np.frombuffer(depth_buf, dtype=np.float32)
        depth_im = depth_im.reshape((height, width))
        depth_im = np.flip(depth_im, axis=0)

        return depth_im

    def _read_main_framebuffer(self, scene, flags):
        width, height = self._main_fb_dims[0], self._main_fb_dims[1]

        # Bind framebuffer and blit buffers
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb_ms)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
        glBlitFramebuffer(
            0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR
        )
        glBlitFramebuffer(
            0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST
        )
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb)

        depth_buf = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
        depth_im = np.frombuffer(depth_buf, dtype=np.float32)
        depth_im = depth_im.reshape((height, width))
        depth_im = np.flip(depth_im, axis=0)

        if sys.platform == "darwin":
            depth_im = self._resize_image(depth_im)

        if flags & RenderFlags.DEPTH_ONLY:
            return depth_im

        # Read color
        if flags & RenderFlags.RGBA:
            color_buf = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
            color_im = np.frombuffer(color_buf, dtype=np.uint8)
            color_im = color_im.reshape((height, width, 4))
        else:
            color_buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            color_im = np.frombuffer(color_buf, dtype=np.uint8)
            color_im = color_im.reshape((height, width, 3))
        color_im = np.flip(color_im, axis=0)

        # Resize for macos if needed
        if sys.platform == "darwin":
            color_im = self._resize_image(color_im, True)

        return color_im, depth_im


class OffRender(OffscreenRenderer):
    def _create(self):
        super()._create()
        self._renderer = MyRender(self.viewport_width, self.viewport_height)
