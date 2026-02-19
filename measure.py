# astr_EF.obj - ectal facet of the astragulus
# calc_EF.obj - ectal facet of the calcaneus
# need to calculate distance between the ectal facets of these bones

import numpy as np

def load_vertices(file_path):

    vertices = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                coordinates = line.split()
                vertices.append([float(coordinates[1]), float(coordinates[2]), float(coordinates[3])])
    
    return np.array(vertices)

# Load your specific facet meshes
astr_vertices = load_vertices('data/astr_EF.obj')
calc_vertices = load_vertices('data/calc_EF.obj')

print(f"Astragalus Facet Vertices: {astr_vertices.shape}")
print(f"Calcaneus Facet Vertices: {calc_vertices.shape}")