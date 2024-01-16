# Standard
# External
import numpy as np
import scipy
from scipy.io import loadmat
# Local


def simplify_array(arr):
    if isinstance(arr, np.ndarray):
        if arr.shape == (1, 1):
            return simplify_array(arr.item())
        else:
            simplified = np.empty(arr.shape, dtype=object)
            for index in np.ndindex(arr.shape):
                simplified[index] = simplify_array(arr[index])
            return simplified
    else:
        return arr

def load_data(datapath):
    raw_data = loadmat(datapath)
    g = raw_data['g'][0][0]
   
    # The first entry is a normal numpy array with N entries
    x = g[0]

    # The second entry is an array of N tuples of 7 arrays with different dtypes
    edges = []
    b = g[1]
    for b_list in b:
        edges_dict = {
            "type": None,
            "from": None,
            "to": None,
            "measurement": None,
            "inf_matrix": None,
            "from_idx": None,
            "to_idx": None
        }

        b_tuple = b_list[0]
        # item 0 is a char - <type>
        b_0 = b_tuple[0].item()
        edges_dict["type"] = b_0
        # item 1, 2 are integers - <from>, <to>
        b_1 = b_tuple[1].item()
        edges_dict["from"] = int(b_1) - 1
        b_2 = b_tuple[2].item()
        edges_dict["to"] = int(b_2) - 1
        # item 3 is a 3x1 array - <measurement> 
        b_3 = b_tuple[3]
        edges_dict["measurement"] = b_3
        # item 4 is a 3x3 array - <information>
        b_4 = b_tuple[4]
        edges_dict["information"] = b_4
        # item 5, 6 are integers - <fromIdx>, <toIdx>
        b_5 = b_tuple[5].item()
        edges_dict["fromIdx"] = int(b_5) - 1
        b_6 = b_tuple[6].item()
        edges_dict["toIdx"] = int(b_6) - 1

        edges.append(edges_dict)

    # The third entry is an array of 1 tuple of N arrays of 2 arrays
    idLookup = [] 
    c = g[2][0]
    c_info = c.dtype
    c_info = np.array([c_info.fields[field][0] for field in c_info.names], dtype=object)
    c_data = c[0]
    for c_tuple_array in c_data:
        idLookup_dict = {
            "offset": [],
            "dimension": []
        }

        c_array = c_tuple_array[0][0]
        # item 0 is a 1x1 array - <offset>
        c_0 = c_array[0].item()
        idLookup_dict["offset"] = int(c_0)
        # item 1 is a 1x1 array - <dimension>
        c_1 = c_array[1].item()
        idLookup_dict["dimension"] = int(c_1)

        idLookup.append(idLookup_dict)

    return x, edges, idLookup
    