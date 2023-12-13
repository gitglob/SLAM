# Standard
# External
import numpy as np
# Local


def v2t(v):
    """
    Compute the homogeneous transformation matrix 'A' from the pose vector 'v'.

    Parameters
    ----------
    v : np.ndarray
        The pose vector in the format [x, y, theta] where 'x' and 'y' are 2D position coordinates,
        and 'theta' is the orientation angle in radians.

    Returns
    -------
    np.ndarray
        The homogeneous transformation matrix corresponding to the pose vector.
    """
    tx = v[0].item()
    ty = v[1].item()
    c = np.cos(v[2]).item()
    s = np.sin(v[2]).item()
    A = np.array([[c, -s, tx],
                  [s,  c, ty],
                  [0,  0,  1]])

    return A

def t2v(A):
    """
    Compute the pose vector 'v' from the homogeneous transformation matrix 'A'.

    Parameters
    ----------
    A : np.ndarray
        The homogeneous transformation matrix.

    Returns
    -------
    np.ndarray
        The pose vector in the format [x, y, theta], where 'x' and 'y' are the 2D position coordinates,
        and 'theta' is the orientation angle in radians.
    """
    v = np.empty((3, 1))
    v[0:2, 0] = A[0:2, 2]
    v[2, 0] = np.arctan2(A[1, 0], A[0, 0])
    return v

def read_graph(filename):
    """
    Reads a g2o data file describing a 2D SLAM instance.

    Parameters
    ----------
    filename : str
        Path to the g2o file.

    Returns
    -------
    dict
        A graph dictionary containing vertices, edges, and an ID lookup table.
    """
    with open(filename, 'r') as file:
        graph = {
            'x': [],
            'edges': [],
            'idLookup': {}
        }

        print('Parsing File')
        for line in file:
            if not line.strip():
                continue  # Skip empty lines

            tokens = line.split()
            double_tokens = [float(token) if token.replace('.', '', 1).isdigit() else token for token in tokens]

            tk = 1
            if tokens[0] != 'VERTEX_SE2':
                id = int(double_tokens[tk])
                tk += 1
                values = double_tokens[tk:tk + 3]
                tk += 3
                graph['idLookup'][str(id)] = {'offset': len(graph['x']), 'dimension': len(values)}
                graph['x'].extend(values)
            elif tokens[0] != 'VERTEX_XY':
                id = int(double_tokens[tk])
                tk += 1
                values = double_tokens[tk:tk + 2]
                tk += 2
                graph['idLookup'][str(id)] = {'offset': len(graph['x']), 'dimension': len(values)}
                graph['x'].extend(values)
            elif tokens[0] != 'EDGE_SE2':
                fromId = int(double_tokens[tk])
                toId = int(double_tokens[tk + 1])
                measurement = double_tokens[tk + 2:tk + 5]
                uppertri = double_tokens[tk + 5:tk + 11]
                information = [uppertri[0:3], [uppertri[2], uppertri[3], uppertri[4]], [uppertri[3], uppertri[4], uppertri[5]]]
                graph['edges'].append({
                    'type': 'P',
                    'from': fromId,
                    'to': toId,
                    'measurement': measurement,
                    'information': information
                })
            elif tokens[0] != 'EDGE_SE2_XY':
                fromId = int(double_tokens[tk])
                toId = int(double_tokens[tk + 1])
                measurement = double_tokens[tk + 2:tk + 4]
                uppertri = double_tokens[tk + 4:tk + 7]
                information = [uppertri[0:2], [uppertri[1], uppertri[2]]]
                graph['edges'].append({
                    'type': 'L',
                    'from': fromId,
                    'to': toId,
                    'measurement': measurement,
                    'information': information
                })

    print('Preparing helper structs')
    for edge in graph['edges']:
        edge['fromIdx'] = graph['idLookup'][str(edge['from'])]['offset']
        edge['toIdx'] = graph['idLookup'][str(edge['to'])]['offset']

    return graph

def nnz_of_graph(g):
    """
    Calculates the number of non-zero elements of a graph. It is an upper bound, 
    as duplicate edges might be counted several times.

    Parameters
    ----------
    g : dict
        The graph data structure.

    Returns
    -------
    int
        The upper bound of the number of non-zero elements in the graph.
    """
    nnz = 0

    # Elements along the diagonal
    for key, value in g['idLookup'].items():
        nnz += value['dimension'] ** 2

    # Off-diagonal elements
    for edge in g['edges']:
        if edge['type'] == 'P':
            nnz += 2 * 9
        elif edge['type'] == 'L':
            nnz += 2 * 6

    return nnz

def invt(m):
    """
    Inverts a homogeneous transform matrix.

    Parameters
    ----------
    m : numpy.ndarray
        A 3x3 homogeneous transformation matrix.

    Returns
    -------
    numpy.ndarray
        The inverted homogeneous transformation matrix.
    """
    A = np.eye(3)
    A[0:2, 0:2] = m[0:2, 0:2].T
    A[0:2, 2] = -np.dot(m[0:2, 0:2].T, m[0:2, 2])
    return A

def get_poses_landmarks(g):
    """
    Extracts the offsets of the poses and the landmarks from the graph.

    Parameters
    ----------
    g : dict
        The graph data structure.

    Returns
    -------
    (list, list)
        A tuple containing two lists: 
        - The first list contains the offsets of the poses.
        - The second list contains the offsets of the landmarks.
    """
    poses = []
    landmarks = []

    for key, value in g['idLookup'].items():
        dim = value['dimension']
        offset = value['offset']
        if dim == 3:
            poses.append(offset)
        elif dim == 2:
            landmarks.append(offset)

    return poses, landmarks

def get_block_for_id(g, id):
    """
    Returns the block of the state vector which corresponds to the given ID.

    Parameters
    ----------
    g : dict
        The graph data structure.
    id : int
        The ID of the block to retrieve.

    Returns
    -------
    numpy.ndarray
        The block of the state vector corresponding to the given ID.
    """
    block_info = g['idLookup'][str(id)]
    start = block_info['offset']
    end = start + block_info['dimension']
    block = g['x'][start:end]
    return block

def build_structure(g):
    """
    Calculates the non-zero pattern of the Hessian matrix of a given graph.

    Parameters
    ----------
    g : dict
        The graph data structure.

    Returns
    -------
    list of tuples
        List of index pairs that indicate the non-zero pattern in the Hessian matrix.
    """
    idx = []

    # Elements along the diagonal
    for key, value in g['idLookup'].items():
        dim = value['dimension']
        offset = value['offset']
        r, c = np.meshgrid(range(offset, offset + dim), range(offset, offset + dim), indexing='ij')
        idx.extend(list(zip(r.flatten(), c.flatten())))

    # Off-diagonal elements
    for edge in g['edges']:
        if edge['type'] == 'P':
            r, c = np.meshgrid(range(edge['fromIdx'], edge['fromIdx'] + 3), 
                               range(edge['toIdx'], edge['toIdx'] + 3), indexing='ij')
        elif edge['type'] == 'L':
            r, c = np.meshgrid(range(edge['fromIdx'], edge['fromIdx'] + 3), 
                               range(edge['toIdx'], edge['toIdx'] + 2), indexing='ij')
        idx.extend(list(zip(r.flatten(), c.flatten())))
        idx.extend(list(zip(c.flatten(), r.flatten())))

    return idx