from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeomNode, Shader, \
    Camera

from sources.mesh import Mesh


class MouseController:
    def __init__(self, camera: Camera, base: ShowBase):
        self.camera = camera
        self.base = base

    def zoom(self):
        self.base.accept('wheel_up', lambda: self.camera.setY(self.camera, -1))
        self.base.accept('wheel_down', lambda: self.camera.setY(self.camera, 1))

    def pan(self):
        pass


class Render(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)

        mesh = Mesh.create_plate(position=(0, 0),
                                 width=6,
                                 height=2,
                                 num_elements_width=6,
                                 num_elements_height=3)
        geom = mesh.get_render_mesh()

        node = GeomNode('g-node')
        node.addGeom(geom)

        node_path = self.render.attachNewNode(node)

        # noinspection PyArgumentList
        shader = Shader.load(lang=Shader.SL_GLSL,
                             vertex="../shaders/grid_shader.vert",
                             fragment="../shaders/grid_shader.frag")
        node_path.setShader(shader)
        node_path.set_shader_input('color', (0.2, 0.5, 0))

        self.camera.setPos(0, 0, 10)
        self.camera.lookAt(node_path)

        self.disableMouse()
        controller = MouseController(self.camera, self)
        controller.zoom()


app = Render()
app.run()
