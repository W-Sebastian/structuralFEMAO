from enum import Enum

from panda3d.core import GeomVertexArrayFormat, Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter, GeomTriangles


class Coordinate:
    def __init__(self, x=0., y=0., z=0.):
        self.coordinate = (x, y, z)

    @property
    def x(self):
        return self.coordinate[0]

    @property
    def y(self):
        return self.coordinate[1]

    @property
    def z(self):
        return self.coordinate[2]


class ElementTopology(Enum):
    Tria3 = 0,
    Quad4 = 1


class Node:
    def __init__(self, node_id: int, coordinate: Coordinate):
        self.id = node_id
        self.coordinate = coordinate


class Element:
    def __init__(self, element_id: int, node_ids: [int], topology: ElementTopology):
        self.id = element_id
        self.nodes = node_ids
        self.type = topology


class Mesh:
    def __init__(self):
        self.nodes: [Node] = []
        self.elements: [Element] = []

    @staticmethod
    def create_plate(position: (float, float),
                     width: float, height: float,
                     num_elements_width: int, num_elements_height: int):

        node_id = 0
        element_id = 0

        left_corner = (position[0] - width / 2, position[1] - height / 2)
        mesh = Mesh()

        num_nodes_x = num_elements_width + 1
        num_nodes_y = num_elements_height + 1

        element_width = width / num_elements_width
        element_height = height / num_elements_height

        def node_index(x, y):
            return y * num_nodes_x + x

        for j in range(0, num_nodes_y):
            for i in range(0, num_nodes_x):
                node = Node(node_id,
                            Coordinate(left_corner[0] + i * element_width,
                                       left_corner[1] + j * element_height,
                                       0))
                node_id += 1
                mesh.nodes.append(node)
                if i > 0 and j > 0:
                    mesh.elements.append(Element(element_id,
                                                 [node_index(i - 1, j - 1),
                                                  node_index(i, j - 1),
                                                  node_index(i, j),
                                                  node_index(i - 1, j)],
                                                 ElementTopology.Quad4))

        return mesh

    def get_render_mesh(self):
        array = GeomVertexArrayFormat()
        array.addColumn("vertex", 3, Geom.NTFloat32, Geom.CPoint)
        array.addColumn("barycenter", 3, Geom.NTFloat32, Geom.CPoint)
        vertex_format = GeomVertexFormat()
        vertex_format.addArray(array)
        vertex_format = GeomVertexFormat.registerFormat(vertex_format)

        v_data = GeomVertexData('mesh', vertex_format, Geom.UHStatic)
        v_data.setNumRows(len(self.nodes))
        vertex = GeomVertexWriter(v_data, 'vertex')
        barycenter = GeomVertexWriter(v_data, 'barycenter')

        prim = GeomTriangles(Geom.UHStatic)
        vertex_id = 0
        for elem in self.elements:
            def node_coords(e, i): return self.nodes[e.nodes[i]].coordinate

            vertex.addData3(node_coords(elem, 0).x, node_coords(elem, 0).y, 0)
            vertex.addData3(node_coords(elem, 1).x, node_coords(elem, 1).y, 0)
            vertex.addData3(node_coords(elem, 2).x, node_coords(elem, 2).y, 0)
            barycenter.addData3(1, 0, 0)
            barycenter.addData3(0, 1, 0)
            barycenter.addData3(0, 0, 1)
            prim.addVertices(vertex_id, vertex_id + 1, vertex_id + 2)
            vertex_id += 3

            vertex.addData3(node_coords(elem, 2).x, node_coords(elem, 2).y, 0)
            vertex.addData3(node_coords(elem, 3).x, node_coords(elem, 3).y, 0)
            vertex.addData3(node_coords(elem, 0).x, node_coords(elem, 0).y, 0)
            barycenter.addData3(1, 0, 0)
            barycenter.addData3(0, 1, 0)
            barycenter.addData3(0, 0, 1)
            prim.addVertices(vertex_id, vertex_id + 1, vertex_id + 2)
            vertex_id += 3

        geom = Geom(v_data)
        geom.addPrimitive(prim)
        return geom
