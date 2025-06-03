from model import LaMO_Regular_Grid, LaMO_Structured_Mesh_2D_shared, LaMO_Structured_Mesh_2D_unshared, LaMO_Irregular_Mesh_shared, LaMO_Irregular_Mesh_unshared


def get_model(args):
    model_dict = {
        'LaMO_Regular_Grid': LaMO_Regular_Grid,    # for regular grid with 4D scan
        'LaMO_Structured_Mesh_2D_shared': LaMO_Structured_Mesh_2D_shared, # for 2D structured mesh with shared enc-dec weights
        'LaMO_Structured_Mesh_2D_unshared': LaMO_Structured_Mesh_2D_unshared, # for 2D structured mesh with unshared enc-dec weights
        'LaMO_Irregular_Mesh_shared': LaMO_Irregular_Mesh_shared, # for irregular mesh with shared enc-dec weights
        'LaMO_Irregular_Mesh_unshared': LaMO_Irregular_Mesh_unshared, # for irregular mesh with unshared enc-dec weights
    }
    return model_dict[args.model]
