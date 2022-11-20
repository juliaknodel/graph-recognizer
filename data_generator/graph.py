from collections import defaultdict
from itertools import combinations
import networkx as nx
import numpy as np
import pygraphviz as pgv
import random
import string

from add_data import INCH_DPI_RATIO, INCH_POINT_RATIO, \
    SMALL_SHIFT, NODE_BBOX_EXP, SHIFT, SHIFT_FROM_BORDER, MAP_SHIFT


def AGraph(directed=False, strict=True, name='', **args):
    """fixed AGraph constructor cuz its broken af"""

    graph = '{0} {1} {2} {{}}'.format(
        'strict' if strict else '',
        'digraph' if directed else 'graph',
        name
    )

    return pgv.AGraph(graph, dpi=INCH_DPI_RATIO, **args)


def drawGraph(adj_list: dict, filename_prefix: str, col=False):
    """
    :param adj_list: graph adjacency list
    :param filename_prefix: filename of the file to save graph image
    :param col: colored graph format or black&white
    :return: pygraphviz AGraph
    """

    g = AGraph(directed=False)
    vertices_num = len(adj_list.keys())

    # if draw edge 2 times -> the number of lines will increase
    # so its necessary to check what has been already rendered
    adj_matrix = np.zeros((vertices_num, vertices_num))

    vertices = adj_list.keys()
    # dict node: number
    vert_ind = dict(zip(vertices, range(0, len(vertices))))

    g.node_attr['shape'] = 'circle'
    g.node_attr['fontname'] = 'Arial'

    if col:
        g.node_attr['fontcolor'] = 'red'

    # ver and ver_2 - vertices connected by edge
    for ver in adj_list.keys():
        for ver_2 in adj_list[ver]:

            # if wasnt rendered
            if adj_matrix[vert_ind[ver]][vert_ind[ver_2]] == 0:

                edge_type = adj_list[ver][ver_2]['type']

                if edge_type == 1:
                    if col:
                        g.add_edge(ver, ver_2,
                                   color='green')
                    else:
                        g.add_edge(ver, ver_2)
                else:
                    if col:
                        g.add_edge(ver, ver_2,
                                   color="blue")
                    else:
                        g.add_edge(ver, ver_2,
                                   color="black:invis:invis:invis:black")

                # add rendered edge to adj_matrix
                adj_matrix[vert_ind[ver]][vert_ind[ver_2]] = 1
                adj_matrix[vert_ind[ver_2]][vert_ind[ver]] = 1

    # draw graph
    g.draw(filename_prefix + ".png", prog="sfdp")
    g.layout(prog="sfdp")

    return g


def generate_names():
    """
    :return: list with all possible node names
    """
    alphabet = [*string.ascii_uppercase]
    names = [a + b for a, b in [*combinations(alphabet, 2)]]

    return names


def get_node_name(node_id: int, node_name_dict: dict, names: list):
    """
    :param node_id: unique node id
    :param node_name_dict: dict with prev assigned names
    :param names: list with all possible names
    :return: node name (assigned earlier or new)
    """

    # if name was assigned earlier -> return it
    if node_id in node_name_dict.keys():
        return node_name_dict, node_name_dict[node_id]

    # free name idx
    node_num = len(node_name_dict)

    # if names are over
    if node_num > len(names) - 1:
        return -1

    # get new name and assign it to node
    new_name = names[node_num]
    node_name_dict[node_id] = new_name

    return node_name_dict, new_name


def get_nodes_in_radius(nodes):
    """
    :param nodes: all nodes
    :return: random node and all nodes nearby
    """

    # random node
    node_num = random.choice([*nodes.keys()])

    # get data from node
    base_node = nodes[node_num]
    base_lat = base_node['lat']
    base_lon = base_node['lon']

    # get all nodes nearby
    nodes_in_neighbourhood = [
        identifier for identifier, data in nodes.items()
        if
        abs(data['lat'] - base_lat) < MAP_SHIFT
        and
        abs(data['lon'] - base_lon) < MAP_SHIFT
    ]

    return nodes_in_neighbourhood


def create_adj_list(ways, friends):
    """
    :param ways: ways from osm
    :param friends: random nodes that are nearby
    :return: adjacency list (format described in readme.md)
    """

    # dictionary to save nodes names by their id
    node_name_dict = dict()
    adj_list = defaultdict(dict)

    # list of possible names
    names = generate_names()

    # searching nodes in all ways
    for way in ways:

        # get nodes connected by this way
        nodes_list = way['nodes']

        for node_num in range(len(nodes_list) - 1):

            # connected by edge nodes
            first_node_id = nodes_list[node_num]
            second_node_id = nodes_list[node_num + 1]

            # if this nodes in friends -> add to graph
            if first_node_id in friends and second_node_id in friends:

                new_name_data = get_node_name(first_node_id,
                                              node_name_dict, names)

                # the names are over
                if new_name_data == -1:
                    print(node_num)
                    return 'More nodes than was expected'

                node_name_dict, first_node_name = new_name_data

                new_name_data = get_node_name(nodes_list[node_num + 1],
                                              node_name_dict, names)
                if new_name_data == -1:
                    print(node_name_dict)
                    return 'More nodes than was expected'

                node_name_dict, second_node_name = new_name_data

                # generate random edge type
                edge_type = random.randint(0, 1)

                # add edge to adj_list,
                # 'weight' = 1 is const & still here only for format purposes
                adj_list[first_node_name][second_node_name] = \
                    {'weigth': '1', 'type': edge_type}
                adj_list[second_node_name][first_node_name] = \
                    {'weigth': '1', 'type': edge_type}

    return adj_list


def is_connected(adj_list: dict):
    """
    :param adj_list: adjacency list
    :return: Bool graph connected
    """

    # its easier to construct a networkx graph
    # and use the function implemented there
    G = nx.Graph()

    for ver in adj_list.keys():

        for ver_2 in adj_list[ver]:
            G.add_edge(ver, ver_2)

    return nx.is_connected(G)


def get_node_boundbox(node, img_x: int, img_y: int):
    """
    :param node: node whose boundbox will be returned
    :param img_x: image width (ox length)
    :param img_y: image height (oy length)
    :return: list with 2 boundbox coordinates [x1, y1, x2, y2]
    """

    pos_x, pos_y = [float(i) * (INCH_DPI_RATIO / INCH_POINT_RATIO)
                    for i in node.attr['pos'].split(',')]

    # node height
    h = float(node.attr['height']) * INCH_DPI_RATIO

    # node width
    w = float(node.attr['width']) * INCH_DPI_RATIO

    # corner_pos = center_pos +- 0.5*width
    x1, x2 = int(pos_x - 0.5 * w), int(pos_x + 0.5 * w)

    # corner_pos = center_pos +- 0.5*height
    y1, y2 = img_y - (int(pos_y - 0.5 * h)), img_y - (int(pos_y + 0.5 * h))

    # expanding bbox borders and check if ox coordinates in [1, img_x - 1]
    # and y_coordinates in [1, img_y - 1]
    # +  shift by SHIFT_FROM_BORDER for visualisation purposes
    x1 = min(max(SHIFT_FROM_BORDER,
                 int(x1) - NODE_BBOX_EXP),
             img_x - SHIFT_FROM_BORDER)
    x2 = min(max(SHIFT_FROM_BORDER,
                 int(x2) + NODE_BBOX_EXP),
             img_x - SHIFT_FROM_BORDER)
    y1 = min(max(SHIFT_FROM_BORDER,
                 int(y1) + NODE_BBOX_EXP),
             img_y - SHIFT_FROM_BORDER)
    y2 = min(max(SHIFT_FROM_BORDER,
                 int(y2) - NODE_BBOX_EXP),
             img_y - SHIFT_FROM_BORDER)

    return [x1, y1, x2, y2]


def get_edge_boundbox(graph: AGraph, edge, img_x: int, img_y: int):
    """
    :param graph: pygraphviz AGraph
    :param edge: edge whose boundbox will be returned
    :param img_x: image width (ox length)
    :param img_y: image height (oy length)
    :return: list with 2 boundbox coordinates [x1, y1, x2, y2]
    """
    node_1, node_2 = edge

    pos_x_1, pos_y_1 = [int(float(i) * (INCH_DPI_RATIO / INCH_POINT_RATIO))
                        for i in graph.get_node(node_1).attr['pos'].split(',')]

    pos_x_2, pos_y_2 = [int(float(i) * (INCH_DPI_RATIO / INCH_POINT_RATIO))
                        for i in graph.get_node(node_2).attr['pos'].split(',')]

    pos_y_1 = img_y - pos_y_1
    pos_y_2 = img_y - pos_y_2

    # if edge horizontal/vertical (detected by small_shift check)
    # -> move coordinates a lil
    if abs(pos_x_1 - pos_x_2) <= SMALL_SHIFT:
        pos_x_1 -= SHIFT
        pos_x_2 += SHIFT
    if abs(pos_y_1 - pos_y_2) <= SMALL_SHIFT:
        pos_y_1 -= SHIFT
        pos_y_2 += SHIFT

    # check if ox coordinates in [1, img_x - 1]
    # and oy coordinates in [1, img_y - 1]
    # SHIFT_FROM_BORDER from image borders for visualisation purposes
    pos_x_1 = min(max(SHIFT_FROM_BORDER,
                      int(pos_x_1)),
                  img_x - SHIFT_FROM_BORDER)
    pos_x_2 = min(max(SHIFT_FROM_BORDER,
                      int(pos_x_2)),
                  img_x - SHIFT_FROM_BORDER)
    pos_y_1 = min(max(SHIFT_FROM_BORDER,
                      int(pos_y_1)),
                  img_y - SHIFT_FROM_BORDER)
    pos_y_2 = min(max(SHIFT_FROM_BORDER,
                      int(pos_y_2)),
                  img_y - SHIFT_FROM_BORDER)

    return [pos_x_1, pos_y_1, pos_x_2, pos_y_2]


def get_edge_type(edge, adj_list: dict):
    """
    :param edge: edge whose type will be returned
    :param adj_list: graph adjacency list
    :return: edge type - 'edge_1' or 'edge_0'
    """

    node_1, node_2 = edge

    return 'edge_1' if adj_list[node_1][node_2]['type'] else 'edge_0'
