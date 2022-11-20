# objects categories
cat = {
    'node': 0,  # node
    'edge_0': 1,  # edge type 0
    'edge_1': 2  # edge type 1
}


# .dot format came straight from the hell,
# so you need to use this constants to convert points
INCH_POINT_RATIO = 72
INCH_DPI_RATIO = 96

# shift for horizontal/vertical edge bbox
SHIFT = 30
# if x1, x2 or y1, y2 of edge bbox is closer than this
# -> its horizontal/vertical edge
SMALL_SHIFT = 10

# node bbox expanded by this value -
# only for my mental health purposes
NODE_BBOX_EXP = 10
# why not
SHIFT_FROM_BORDER = 1
# minutes of experience and i got this value
# (for searhing nodes nearby)
MAP_SHIFT = 0.001
