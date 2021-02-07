from enum import Enum

from scipy.linalg import qr

from mesh import Mesh, Node, Coordinate
import numpy as np


def solve_mldivide(A, b):
    x1, res, rnk, s = np.linalg.lstsq(A, b, rcond=None)
    if rnk == A.shape[1]:
        return x1  # nothing more to do if A is full-rank
    Q, R, P = qr(A.T, mode='full', pivoting=True)
    Z = Q[:, rnk:].conj()
    C = np.linalg.solve(Z[rnk:], -x1[rnk:])
    return x1 + Z.dot(C)


class NaturalCoordinate:
    def __init__(self, x=0., y=0., z=0.):
        self.coordinate = (x, y, z)

    @property
    def r(self):
        return self.coordinate[0]

    @property
    def s(self):
        return self.coordinate[1]

    @property
    def t(self):
        return self.coordinate[2]


class ShapeFunctionDerivative:
    def __init__(self, dr=0., ds=0., dt=0.):
        self.derivative = (dr, ds, dt)

    @property
    def dr(self):
        return self.derivative[0]

    @property
    def ds(self):
        return self.derivative[1]

    @property
    def dt(self):
        return self.derivative[2]


class GaussPoint:
    def __init__(self, coordinate: NaturalCoordinate, weight: float):
        self.coordinate = coordinate
        self.weight = weight


class PointForceLoad:
    def __init__(self, nodes: [int], value: [float]):
        self.nodes = nodes
        self.value = value


class FixedConstraint:
    def __init__(self, nodes: [int], value: [float]):
        self.nodes = nodes
        self.value = value


natural_coordinates_quad_4 = [
    NaturalCoordinate(-1, -1, 0),
    NaturalCoordinate(+1, -1, 0),
    NaturalCoordinate(+1, +1, 0),
    NaturalCoordinate(-1, +1, 0)
]


def shape_function_quad_4(coord: NaturalCoordinate):
    r = coord.r
    s = coord.s
    aux = r * s
    return [
        (1 - r - s + aux) / 4,
        (1 + r - s - aux) / 4,
        (1 + r + s + aux) / 4,
        (1 - r + s - aux) / 4
    ]


def shape_function_derivative_quad_4(coord: NaturalCoordinate):
    r = coord.r
    s = coord.s
    return [
        ShapeFunctionDerivative(dr=(-1 + s) * 0.25, ds=(-1 + r) * 0.25),
        ShapeFunctionDerivative(dr=(+1 - s) * 0.25, ds=(-1 - r) * 0.25),
        ShapeFunctionDerivative(dr=(+1 + s) * 0.25, ds=(+1 + r) * 0.25),
        ShapeFunctionDerivative(dr=(-1 - s) * 0.25, ds=(+1 - r) * 0.25)
    ]


def gauss_quadrature_quad_4():
    return [
        GaussPoint(NaturalCoordinate(-0.577350269189626, -0.577350269189626, 0), weight=1.0),
        GaussPoint(NaturalCoordinate(-0.577350269189626, +0.577350269189626, 0), weight=1.0),
        GaussPoint(NaturalCoordinate(+0.577350269189626, -0.577350269189626, 0), weight=1.0),
        GaussPoint(NaturalCoordinate(+0.577350269189626, +0.577350269189626, 0), weight=1.0)
    ]


class PlaneAnalysisType(Enum):
    PlaneStress = 0,
    PlaneStrain = 1


def jacobian_at_coordinate(nodes: [Node], natural_coordinate: NaturalCoordinate):
    sfd = shape_function_derivative_quad_4(natural_coordinate)

    jacobianS = np.zeros([3])
    jacobianR = np.zeros([3])
    jacobianT = np.zeros([3])

    for node, derivative in zip(nodes, sfd):
        dr = derivative.dr
        ds = derivative.ds

        nc = np.array([node.coordinate.x, node.coordinate.y, node.coordinate.z])

        jacobianS = np.add(jacobianS, nc * dr)
        jacobianR = np.add(jacobianR, nc * ds)

    jacobianT = np.cross(jacobianS, jacobianR)

    return np.array([jacobianS, jacobianR, jacobianT])


class Solver:
    def __init__(self, mesh: Mesh, forces: [PointForceLoad], constraint: [FixedConstraint]):
        self.mesh = mesh
        self.forces = forces
        self.constraints = constraint

    @staticmethod
    def compute_b_matrix(nodes: [Node], jacobian: np.ndarray, sf_derivative: [ShapeFunctionDerivative]):
        b_matrix = np.zeros([3, 8])

        # shape function local derivatives
        sf_local_derivative = np.array([
            [d.dr for d in sf_derivative],
            [d.ds for d in sf_derivative],
            np.zeros(4)])

        # cartesian derivatives of the shape functions
        cartesian_derivative_sf = solve_mldivide(jacobian, sf_local_derivative)

        for i in range(0, len(nodes)):
            b_matrix[0:3, i * 2:i * 2 + 2] = np.array([
                [cartesian_derivative_sf[0, i], 0],
                [0, cartesian_derivative_sf[1, i]],
                [cartesian_derivative_sf[1, i], cartesian_derivative_sf[0, i]]
            ])

        return b_matrix

    @staticmethod
    def element_stiffness_matrix(element_nodes: [Node]):
        """
        K = [transpose(B) * D * B] * thickness * area
        :return:
        """
        gauss_quadrature = gauss_quadrature_quad_4()
        thickness = 1.0

        D = Solver.element_elastic_matrix()

        M = np.zeros([8, 8])
        for gauss_point in gauss_quadrature:
            sf_derivative_at_gauss_point = shape_function_derivative_quad_4(gauss_point.coordinate)
            jacobian = jacobian_at_coordinate(element_nodes, gauss_point.coordinate)

            B = Solver.compute_b_matrix(element_nodes, jacobian, sf_derivative_at_gauss_point)
            area = np.linalg.norm(jacobian[2]) * gauss_point.weight * thickness
            M = M + np.transpose(B).dot(D).dot(B) * area

        return M

    @staticmethod
    def element_strain_matrix(coordinates):
        """
        B matrix
        :return: element matrix
        """
        matrix = np.zeros((3, len(coordinates) * 2))

        for index, node in enumerate(coordinates):
            drs = shape_function_derivative_quad_4(node)[index]

            matrix[0:3, index * 2:index * 2 + 2] = [
                [drs.dr, 0],
                [0, drs.ds],
                [drs.ds, drs.dr]
            ]
        return matrix

    @staticmethod
    def element_elastic_matrix(young_modulus=1.8e11, poisson_ratio=0.3, analysis=PlaneAnalysisType.PlaneStress):
        """
        D matrix
        :return: elastic matrix
        """
        if analysis == PlaneAnalysisType.PlaneStress:
            aux1 = young_modulus / (1 - poisson_ratio ** 2)
            aux2 = poisson_ratio * aux1
            aux3 = young_modulus / 2 / (1 + poisson_ratio)
        else:
            aux1 = young_modulus * (1 - poisson_ratio) / (1 + poisson_ratio) / (1 - 2 * poisson_ratio)
            aux2 = aux1 * poisson_ratio / (1 - poisson_ratio)
            aux3 = young_modulus / 2 / (1 + poisson_ratio)
        matrix = [[aux1, aux2, 0], [aux2, aux1, 0], [0, 0, aux3]]
        return matrix

    def compute(self):
        # problem size
        num_dofs = len(self.mesh.nodes) * 2
        stiffness_matrix = np.zeros((num_dofs, num_dofs))

        # get the forces (considering only external ones so far)
        force_array = np.zeros(num_dofs)
        for force in self.forces:
            nodes = force.nodes
            for node in nodes:
                eq_number = node*2
                force_array[eq_number] = force.value[0]  # +X
                force_array[eq_number+1] = force.value[1]  # +Y

        # get the displacement array with the known values
        displacements = np.zeros((num_dofs, 1))
        fixed_nodes = []  # also keep the fixed equations from the matrix
        for constraint in self.constraints:
            nodes = constraint.nodes
            for node in nodes:
                eq_number = node*2
                displacements[eq_number] = constraint.value[0]
                displacements[eq_number+1] = constraint.value[1]
                # for these equations we don't need to solve the system
                fixed_nodes.extend([eq_number, eq_number + 1])

        # assemble the global stiffness matrix
        for element in self.mesh.elements:
            node_ids = element.nodes
            element_nodes = [self.mesh.nodes[i] for i in node_ids]

            Ke = self.element_stiffness_matrix(element_nodes)

            num_element_eq = len(node_ids) * 2
            eq_number = []
            for x in node_ids:
                eq_number.extend([2 * x, 2 * x + 1])

            for i in range(0, num_element_eq):
                for j in range(0, num_element_eq):
                    stiffness_matrix[eq_number[i], eq_number[j]] = \
                        stiffness_matrix[eq_number[i], eq_number[j]] + Ke[i, j]

        reduced_stiffness_matrix = np.delete(stiffness_matrix, fixed_nodes, 0)
        reduced_stiffness_matrix = np.delete(reduced_stiffness_matrix, fixed_nodes, 1)

        reduced_forces = np.delete(force_array, fixed_nodes, 0)

        print(reduced_stiffness_matrix.shape)
        print(reduced_forces.shape)
        u = solve_mldivide(reduced_stiffness_matrix, reduced_forces)
        print(u)


# nodes = [
#     Node(1, Coordinate(0, 0, 0)),
#     Node(2, Coordinate(0, 0, 6)),
#     Node(3, Coordinate(0, 2, 6)),
#     Node(4, Coordinate(0, 2, 0))
# ]


test_nodes = [
    Node(0, Coordinate(0, 0, 0)),
    Node(1, Coordinate(2, 0, 0)),
    Node(2, Coordinate(2, 6, 0)),
    Node(3, Coordinate(0, 6, 0)),
]

input_mesh = Mesh.create_plate(position=(0, 0),
                               width=60e-3,
                               height=20e-3,
                               num_elements_width=2,
                               num_elements_height=1)

input_forces = [PointForceLoad([0, 3], [-100, 0])]
input_constraints = [FixedConstraint([5, 2], [0, 0])]

Solver(input_mesh, input_forces, input_constraints).compute()
