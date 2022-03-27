import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import Misc_Folder.Constants as c


class Network:
    def __init__(self, opt, rings_number):
        if rings_number == 0:
            target_dict = {}
            for j in range(1, opt.nodes_number + 1):
                target_dict[j] = {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}
            self.edges_attr = {}
            for i in range(0, opt.nodes_number):
                if i == 0:
                    self.edges_attr[int(i)] = target_dict
                else:
                    self.edges_attr[int(i)] = {}
            self.edges_attr[3] = {5:{'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}}
            del self.edges_attr[0][5]
        if rings_number == 5:
            self.edges_attr = {
                0: {1: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                    2: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                    3: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                1: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                    2: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, },
                2: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                    3: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                    1: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                3: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                    2: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
            }
        if rings_number == 10:
            self.edges_attr = {
                0: {1: {'cost': 1, 'weight': 10}},
                1: {2: {'cost': 1, 'weight': 20}},
                2: {0: {'cost': 1, 'weight': 30}}
            }
        if rings_number == 1:
            self.edges_attr = {
                0: {1: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 4: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 5: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                1: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 2: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 5: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                2: {1: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 6: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 3: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                3: {2: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 6: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 7: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                4: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 8: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 5: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                5: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 1: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 4: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                6: {2: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 3: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 7: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                7: {3: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 6: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 11: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                8: {4: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 9: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 12: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                9: {8: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 12: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 13: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                10: {11: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 14: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 15: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                11: {10: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 15: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 7: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                12: {8: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 9: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 13: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                13: {9: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 12: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 14: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                14: {10: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 13: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 15: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                15: {10: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 11: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 14: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
            }
        if rings_number == 2:
            self.edges_attr = {0: {1: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 2: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 3: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   4: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 5: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 6: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               1: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 7: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 8: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   9: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               2: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 9: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 10: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   11: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               3: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 11: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 12: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   13: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               4: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 13: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 14: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   15: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               5: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 15: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 16: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   17: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               6: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 17: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 18: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   7: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               7: {6: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 1: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               8: {1: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               9: {1: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 2: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               10: {2: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               11: {2: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 3: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               12: {3: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               13: {3: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 4: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               14: {4: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               15: {4: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 5: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               16: {5: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               17: {5: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 6: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               18: {6: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               }
        if rings_number == 3:
            self.edges_attr = {0: {1: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 2: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 3: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   4: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 5: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 6: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               1: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 7: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 8: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   9: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               2: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 9: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 10: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   11: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               3: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 11: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 12: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   13: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               4: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 13: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 14: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   15: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               5: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 15: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 16: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   17: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               6: {0: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 17: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 18: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   7: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               7: {19: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 20: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 6: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   1: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               8: {20: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 21: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   1: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               9: {21: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 22: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 1: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                   2: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               10: {22: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 23: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                    2: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               11: {23: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 24: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 2: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                    3: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               12: {24: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 25: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                     3: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               13: {25: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 26: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 3: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                    4: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               14: {26: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},27: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 4: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               15: {27: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 28: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 4: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                    5: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               16: {28: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                    29: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 5: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               17: {29: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 30: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 5: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                    6: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               18: {30: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)},
                                    19: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 6: {'cost': 2, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               19: {18: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 7: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               20: {7: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 8: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               21: {8: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 9: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               22: {9: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 10: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               23: {10: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 11: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               24: {11: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 12: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               25: {12: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 13: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               26: {14: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 13: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               27: {14: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 15: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               28: {16: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 15: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               29: {16: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 17: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               30: {17: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}, 18: {'cost': 1, 'weight': np.random.randint(opt.capacity_per_link_low, opt.capacity_per_link_high)}},
                               }

        self.graph_topology = nx.DiGraph(self.edges_attr)
        self.df = nx.to_pandas_edgelist(self.graph_topology)
        edges_dict = {}
        for index, row in self.df.iterrows():
            edges_dict[str(str(row['source']) + "_" + str(row['target']))] = row['weight']
        self.mask_edges = nx.to_pandas_adjacency(self.graph_topology).mask(nx.to_pandas_adjacency(self.graph_topology) > 0, 1)
        self.edges_weight = edges_dict
        self.avg_shortest_path = nx.average_shortest_path_length(self.graph_topology)
        # self.nodes_positions = nx.spectral_layout(self.graph_topology)
        self.nodes_positions = nx.spectral_layout(self.graph_topology)
        nx.draw_spectral(self.graph_topology, with_labels=True)
        plt.show()
        # self.nodes_positions = nx.spring_layout(self.edges_attr)
        logger.info(f"Network was created successfully")

    def network_create(self):
        edge_keys = nx.to_pandas_edgelist(self.graph_topology)
        connectivity_matrix_w = nx.to_pandas_adjacency(self.graph_topology)
        return connectivity_matrix_w, edge_keys

    def network_plot(self):
        # nx.draw_spectral(self.graph_topology, with_labels=True)
        # nx.draw_spring(self.graph_topology, with_labels=True)
        nx.draw_planar(self.graph_topology, with_labels=True)
        D = nx.from_pandas_edgelist(self.df, edge_attr=["weight", "cost"])
        # labels = nx.get_edge_attributes(D, 'weight')
        # nx.draw_networkx_edge_labels(D, nx.shell_layout(self.graph_topology), edge_labels=labels)
        plt.show()
