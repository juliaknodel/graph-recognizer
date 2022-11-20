import cv2
from detectron2.structures import BoxMode
import json
import overpass
import pickle
import os

from tqdm import tqdm

from add_data import cat
from graph import get_nodes_in_radius, create_adj_list, \
    is_connected, drawGraph, get_node_boundbox, get_edge_boundbox, \
    get_edge_type, AGraph


def generate(n_data: int, directory='graph_img_private'):
    """
    function which generates n_data graphs (.png),
    their adj_lists (.pickle) and data for detectron2 custom datasets (.json)

    :param directory: directory to save data
    :param n_data: number of graphs + metadata to generate
    :return:
    """

    # get all ways and nodes connected by them from the map
    ways, nodes = get_map_data()

    if not os.path.isdir(directory):
        os.makedirs(directory)

    # i think its faster to create a new connected graph
    # than split an unconnected one into several connected
    # so we need this counter for image filenames
    cnt = 1

    for _ in tqdm(range(n_data)):

        friends = get_nodes_in_radius(nodes)
        adj_list = create_adj_list(ways, friends)

        filename_prefix = os.path.join(directory, 'graph_' + str(cnt))

        if is_connected(adj_list):
            data_to_pickle(adj_list, filename_prefix)
            G = drawGraph(adj_list, filename_prefix)

            data_to_json(G, filename_prefix,
                         filename_prefix + '.png', adj_list)
            cnt += 1


def data_to_json(graph: AGraph, filename_prefix: str, img_filename: str, adj_list: dict):
    """
    this function generates custom dataset for graph
    and writes it to the filename_prefix.json
    :param graph: pygraphviz AGraph
    :param filename_prefix: filename prefix of file
                            for writing the generated data
    :param img_filename: path to graph image generated earlier
    :param adj_list: custom adjacency list structure for graph
    :return:
    """

    # about custom dataset structure:
    # https://detectron2.readthedocs.io/en/stable/tutorials/datasets.html

    info = {}

    img = cv2.imread(img_filename)
    img_y = len(img)
    img_x = len(img[0])

    # path to the image
    info['filename'] = img_filename

    # image width and height
    info['width'] = img_x
    info['height'] = img_y

    # must be unique
    info['image_id'] = img_filename

    # it'll be list(dict)
    info['annotations'] = []

    # dict for every node
    for node in graph.nodes():
        annotation = {
            'bbox': get_node_boundbox(node, img_x, img_y),  # boundbox
            'bbox_mode': BoxMode.XYXY_ABS,  # 'bbox' data format
            'category_id': cat['node']  # object category
        }

        info['annotations'].append(annotation)

    # dict for every edge
    for edge in graph.edges():
        annotation = {
            'bbox': get_edge_boundbox(graph, edge, img_x, img_y),
            'bbox_mode': BoxMode.XYXY_ABS,
            'category_id': cat[get_edge_type(edge, adj_list)]
        }

        info['annotations'].append(annotation)

    # write generated data to .json
    with open(filename_prefix + '.json', 'w') as fp:
        json.dump(info, fp)


def data_to_pickle(adj_list: dict, filename_prefix: str):
    """
    this function writes adj_list to the filename_prefix.pickle
    :param adj_list: adjacency list
    :param filename_prefix: filename prefix of file
                            for writing the generated data
    :return:
    """

    with open(filename_prefix + '.pickle', 'wb') as handle:
        pickle.dump(adj_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_map_data():
    """
    this function request information about ways
    and nodes connected by them from OSM
    :return: ways, nodes
    """
    api = overpass.API()

    area = 'Санкт-Петербург'

    # overpass has a specific syntax))
    rail_response = api.get(
        f'area[name="{area}"];way(area)[highway=path];(._;>;);',
        responseformat="json"
    )

    # a lot of requests == a lot of failures,
    # so it's better to save the data to a file right away
    with open('response.json', 'w') as rf:
        json.dump(rail_response, rf)

    elements = rail_response['elements']

    # way is a sequence of nodes connected by him
    ways = [el for el in elements
            if 'type' in el and el['type'] == 'way']

    nodes = {el['id']: el for el in elements
             if 'type' in el and el['type'] == 'node'}

    return ways, nodes


if __name__ == '__main__':
    generate(5, '../graph_img_public_1')
