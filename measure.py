# astr_EF.obj - ectal facet of the astragulus
# calc_EF.obj - ectal facet of the calcaneus
# need to calculate distance between the ectal facets of these bones

import numpy as np
import pandas as pd

# method to load vertices
def load_vertices(file_path):
    vertices = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                coordinates = line.split()
                vertices.append([float(coordinates[1]), float(coordinates[2]), float(coordinates[3])])
    
    return np.array(vertices)

astr_vertices = load_vertices('data/astr_EF.obj')
calc_vertices = load_vertices('data/calc_EF.obj')

print(f"Astragalus Facet Vertices: {astr_vertices.shape}")
print(f"Calcaneus Facet Vertices: {calc_vertices.shape}")

astr_transform = pd.read_csv('data/astr_transformations.csv')
calc_transform = pd.read_csv('data/calc_transformations.csv')

print(f"Astragalus Transformations: {astr_transform.shape}")
print(f"Calcaneus Transformations: {calc_transform.shape}")

# method to apply each maya transform
def apply_transformation(vertices, matrix):
    transform_matrix = np.array(matrix).reshape(4,4)

    one = np.ones((vertices.shape[0], 1))
    homogenous_verticies = np.hstack((vertices, one))

    real_vertices = homogenous_verticies @ transform_matrix

    return real_vertices[:, :3]

