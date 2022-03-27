import numpy as np


class Interference:
    def __init__(self, net, edges, edges_dict, opts):
        self.id_to_edges = edges
        self.edges_to_id = edges_dict
        self.capacity_per_edge = {}
        self.opts = opts
        self.net = net
        for i in range(len(self.net.df)):
            self.capacity_per_edge[str(self.net.df.iloc[i, 0]) + "_" + str(self.net.df.iloc[i, 1])] = int(
                str(self.net.df.iloc[i, 3]))

    def make_angle(self, a, b, c):
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def angle_to_power(self, angle):
        if 0 < angle < 90:
            x = 1 - (angle / 90)
        else:
            x = 0
        return x

    def get_node_edges(self, node, dict):
        edges_list = []
        for k in dict.keys():
            if dict[k].split("_")[0] == node or dict[k].split("_")[1] == node:
                edges_list.append(dict[k])
        return edges_list

    def make_interference_model(self, compare, value):
        interfere_matrix = np.zeros(shape=(len(self.net.nodes_positions), len(self.net.nodes_positions)))
        np.fill_diagonal(interfere_matrix, 0)
        dis_matrix = np.zeros(shape=(len(self.net.nodes_positions), len(self.net.nodes_positions)))
        np.fill_diagonal(dis_matrix, 0)
        angle_matrix = np.zeros(shape=(len(self.net.nodes_positions), len(self.net.nodes_positions)))
        np.fill_diagonal(angle_matrix, 0)
        for i in range(len(self.net.nodes_positions)):
            for j in range(len(self.net.nodes_positions)):
                distance = np.linalg.norm(self.net.nodes_positions[i] - self.net.nodes_positions[j])
                dis_matrix[i][j] = dis_matrix[j][i] = distance
                angle_matrix[i][j] = self.make_angle(self.net.nodes_positions[i], [0, 0], self.net.nodes_positions[j])
        np.fill_diagonal(angle_matrix, 0)
        self.distance_matrix = dis_matrix
        self.angles_matrix = angle_matrix
        for i in range(len(self.net.nodes_positions)):
            for j in range(len(self.net.nodes_positions)):
                if i == j:
                    interfere_matrix[i][j] = 0
                    continue
                interfere_matrix[i][j] = float(self.angle_to_power(self.angles_matrix[i][j]) *
                                               np.power(1 / (4 * np.pi * self.distance_matrix[i][j]), 2))
                # the amount of interference that node i causes to node j

        self.interferce_matrix_nodes = interfere_matrix
        self.interference_matrix_edges = np.zeros(shape=(len(self.capacity_per_edge), len(self.capacity_per_edge)))
        self.masked_interfernce_mat_nodes = np.array(
            np.ma.masked_array(self.interferce_matrix_nodes, mask=self.net.mask_edges))
        # edge interfer_matrix_edges[i][j] - edge i is interferring all other edges [j] by value.
        for index_of_edge in self.id_to_edges.keys():
            node_i = int(self.id_to_edges[index_of_edge].split("_")[0])  # i_j
            node_j = int(self.id_to_edges[index_of_edge].split("_")[1])
            first_vector = np.array([self.net.nodes_positions[node_j][0] - self.net.nodes_positions[node_i][0],
                                     self.net.nodes_positions[node_j][1] - self.net.nodes_positions[node_i][1]])
            unit_first_vector = first_vector / np.linalg.norm(first_vector)
            for idx in self.id_to_edges.keys():
                if idx != index_of_edge:
                    node_i_idx = int(self.id_to_edges[idx].split("_")[0])  # i_j
                    node_j_idx = int(self.id_to_edges[idx].split("_")[1])
                    second_vector = np.array(
                        [self.net.nodes_positions[node_j_idx][0] - self.net.nodes_positions[node_i_idx][0],
                         self.net.nodes_positions[node_j_idx][1] - self.net.nodes_positions[node_i_idx][1]])
                    unit_second_vector = second_vector / np.linalg.norm(second_vector)

                    angle = self.make_angle(unit_first_vector, [0, 0], unit_second_vector)
                    if node_i_idx == node_j:
                        interference_value = 0
                    else:
                        interference_value = self.angle_to_power(angle) * np.power(
                            1 / (4 * np.pi * self.distance_matrix[node_i][node_j_idx]), 2)
                    if self.opts.interference == 1 and compare is False:
                        self.interference_matrix_edges[self.edges_to_id[self.id_to_edges[index_of_edge]]][
                            self.edges_to_id[self.id_to_edges[idx]]] = interference_value + np.random.uniform(
                            low=self.opts.low_interference, high=self.opts.high_interference)
                    elif self.opts.interference == 1 and compare is True:   ##EVAL MODE
                        self.interference_matrix_edges[self.edges_to_id[self.id_to_edges[index_of_edge]]][
                            self.edges_to_id[self.id_to_edges[idx]]] = interference_value + np.random.uniform(low=0, high=value)
                    else:##VERBOSE MODE
                        self.interference_matrix_edges[self.edges_to_id[self.id_to_edges[index_of_edge]]][
                            self.edges_to_id[self.id_to_edges[idx]]] = 0.02
                    # interfere_mat_edge[i][j] - edge i is interfering  edge j by the value
        self.interference_matrix_edges = np.nan_to_num(self.interference_matrix_edges, nan=0.0, posinf=1)
        return
