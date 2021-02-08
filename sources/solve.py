from enum import Enum

from scipy.linalg import qr
from scipy.sparse import bsr_matrix, csr_matrix, lil_matrix
from scipy.sparse.linalg import lsqr

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

        # get the forces (considering only external ones so far)
        force_array = np.zeros(num_dofs)
        for force in self.forces:
            nodes = force.nodes
            for node in nodes:
                eq_number = node*2
                force_array[eq_number] = force.value[0]  # +X
                force_array[eq_number+1] = force.value[1]  # +Y

        # get the displacement array with the known values
        displacements = np.zeros(num_dofs)
        fixed_nodes_eq_numbers = []  # also keep the fixed equations from the matrix
        for constraint in self.constraints:
            nodes = constraint.nodes
            for node in nodes:
                eq_number = node*2
                displacements[eq_number] = constraint.value[0]
                displacements[eq_number+1] = constraint.value[1]
                # for these equations we don't need to solve the system
                fixed_nodes_eq_numbers.extend([eq_number, eq_number + 1])

        # assemble the global stiffness matrix
        num_computing_dofs = num_dofs - len(fixed_nodes_eq_numbers)
        stiffness_matrix = lil_matrix((num_computing_dofs, num_computing_dofs))
        # Issue here; because our stiffness matrix already has equations remove indices don't work
        # we build a method to retrieve the actual index
        # map index from range (0, num_dofs) to (0, num_computing_dofs)
        r = iter(range(0, num_computing_dofs))
        dof_mapping = [next(r) if i not in fixed_nodes_eq_numbers else -1 for i in range(0, num_dofs)]

        for element in self.mesh.elements:
            print("element {}".format(element.id))
            node_ids = element.nodes
            element_nodes = [self.mesh.nodes[i] for i in node_ids]

            Ke = self.element_stiffness_matrix(element_nodes)

            num_element_eq = len(node_ids) * 2
            eq_number = []
            for x in node_ids:
                eq_number.extend([2 * x, 2 * x + 1])

            for i in range(0, num_element_eq):
                for j in range(0, num_element_eq):
                    # get the equation numbers from the full list of DOFs
                    eq_idx_x = eq_number[i]
                    eq_idx_y = eq_number[j]
                    # skip it if this is constrained
                    if eq_idx_x in fixed_nodes_eq_numbers or eq_idx_y in fixed_nodes_eq_numbers:
                        continue
                    # change the index to the equations to be solved
                    eq_idx_x = dof_mapping[eq_idx_x]
                    eq_idx_y = dof_mapping[eq_idx_y]

                    stiffness_matrix[eq_idx_x, eq_idx_y] = \
                        stiffness_matrix[eq_idx_x, eq_idx_y] + Ke[i, j]

        # reduced_stiffness_matrix = np.delete(stiffness_matrix, fixed_nodes_eq_numbers, 0)
        # reduced_stiffness_matrix = np.delete(reduced_stiffness_matrix, fixed_nodes_eq_numbers, 1)

        reduced_forces = np.delete(force_array, fixed_nodes_eq_numbers, 0)

        print("Solving the system...")
        u = lsqr(stiffness_matrix, reduced_forces)[0]
        # u = solve_mldivide(reduced_stiffness_matrix, reduced_forces)
        print("Combining results...")
        j = 0
        for i in range(0, displacements.shape[0]):
            if i in fixed_nodes_eq_numbers:
                continue
            displacements[i] = u[j]
            j += 1

        return displacements

    @staticmethod
    def stress_element_computation(element_nodes: [Node], displacements: np.ndarray):
        gauss_quadrature = gauss_quadrature_quad_4()

        D = Solver.element_elastic_matrix()

        extrap = np.zeros((4, 4))
        strsg = np.zeros((3, 4))

        indices = [0, 3, 1, 2]
        current_index = iter(indices)
        for gauss_point in gauss_quadrature:
            idx = next(current_index)
            sf_derivative_at_gauss_point = shape_function_derivative_quad_4(gauss_point.coordinate)
            jacobian = jacobian_at_coordinate(element_nodes, gauss_point.coordinate)

            B = Solver.compute_b_matrix(element_nodes, jacobian, sf_derivative_at_gauss_point)

            strsg[:, idx] = np.dot(np.dot(D, B), displacements)
            shape_functions = shape_function_quad_4(NaturalCoordinate(
                    1/gauss_point.coordinate.r,
                    1/gauss_point.coordinate.s, 0))

            extrap[idx, :] = np.array(shape_functions)

        stress = extrap.dot(strsg.transpose()).transpose()

        return stress

    def stress_computation(self, displacements):

        nodal_stress = np.zeros((len(self.mesh.nodes), 3))
        num_values = np.zeros(len(self.mesh.nodes), dtype=np.uintc)

        for element in self.mesh.elements:
            node_ids = element.nodes
            element_nodes = [self.mesh.nodes[i] for i in node_ids]
            element_displacements = []
            for i in node_ids:
                element_displacements.extend([displacements[i*2], displacements[i*2+1]])
            stresses = self.stress_element_computation(element_nodes, np.array(element_displacements))

            idx = 0
            for i in node_ids:
                nodal_stress[i, :] += stresses[:, idx]
                num_values[i] += 1
                idx += 1

        for idx in range(0, len(self.mesh.nodes)):
            if num_values[idx] != 0:
                nodal_stress[idx, :] /= num_values[idx]

        return nodal_stress

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


def test_case_stress():
    num_elements_x = 2
    num_elements_y = 2
    num_total_nodes = (num_elements_x + 1) * (num_elements_y + 1)

    input_mesh = Mesh.create_plate(position=(0, 0),
                                   width=60e-3,
                                   height=20e-3,
                                   num_elements_width=num_elements_x,
                                   num_elements_height=num_elements_y)

    input_forces_nodes = list(range(0, num_total_nodes, num_elements_x + 1))
    input_forces = [PointForceLoad(
        [input_forces_nodes[3], input_forces_nodes[-1]],
        [-100, 0])]

    input_constraints_nodes = list(range(num_elements_x, num_total_nodes, num_elements_x + 1))
    input_constraints = [FixedConstraint(
        input_constraints_nodes,
        [0, 0])]

    print("Computing...")
    solver = Solver(input_mesh, input_forces, input_constraints)
    u_x_y = solver.stress_element_computation(test_nodes, np.array([1, 2, 3, 4, 5, 6, 7, 8]))


def test_case():
    compute_stress = True

    num_elements_x = 50
    num_elements_y = 50
    num_total_nodes = (num_elements_x + 1) * (num_elements_y + 1)

    input_mesh = Mesh.create_plate(position=(0, 0),
                                   width=60,
                                   height=20,
                                   num_elements_width=num_elements_x,
                                   num_elements_height=num_elements_y)

    input_forces_nodes = list(range(0, num_total_nodes, num_elements_x + 1))
    input_forces = [PointForceLoad(
        input_forces_nodes,
        [-1000, 0])]

    input_constraints_nodes = list(range(num_elements_x, num_total_nodes, num_elements_x + 1))
    input_constraints = [FixedConstraint(
        input_constraints_nodes,
        [0, 0])]

    print("Computing...")
    solver = Solver(input_mesh, input_forces, input_constraints)
    u_x_y = solver.compute()

    print("Writing Results...")
    with open('displacement_x.txt', 'w') as mesh_file:
        for node, u in zip(input_mesh.nodes, u_x_y[0::2]):
            mesh_file.write("{} {} {}\n".format(node.coordinate.x, node.coordinate.y, u))

    with open('displacement_y.txt', 'w') as mesh_file:
        for node, u in zip(input_mesh.nodes, u_x_y[1::2]):
            mesh_file.write("{} {} {}\n".format(node.coordinate.x, node.coordinate.y, u))

    if compute_stress:
        stress = solver.stress_computation(u_x_y)

        print("Writing Stress...")
        with open('stress_x.txt', 'w') as mesh_file:
            for node, u in zip(input_mesh.nodes, stress):
                mesh_file.write("{} {} {}\n".format(node.coordinate.x, node.coordinate.y, u[0]))

        with open('stress_y.txt', 'w') as mesh_file:
            for node, u in zip(input_mesh.nodes, stress):
                mesh_file.write("{} {} {}\n".format(node.coordinate.x, node.coordinate.y, u[1]))
                
        with open('stress_xy.txt', 'w') as mesh_file:
            for node, u in zip(input_mesh.nodes, stress):
                mesh_file.write("{} {} {}\n".format(node.coordinate.x, node.coordinate.y, u[2]))


test_case()
