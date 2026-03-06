# astr_EF.obj - ectal facet of the astragulus
# calc_EF.obj - ectal facet of the calcaneus
# need to calculate distance between the ectal facets of these bones

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

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

astr_vertices = astr_vertices * 0.1
calc_vertices = calc_vertices * 0.1

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
    homogenous_vertices = np.hstack((vertices, one))
    real_vertices = homogenous_vertices @ transform_matrix

    return real_vertices[:, :3]/real_vertices[:, 3:4]

distances = []
frame_count = astr_transform.shape[0]

print("Processing " + str(frame_count) + " frames...")

# iterate through every frame and calculate the distance
for frame in range(frame_count):
    astr_transform_matrix = astr_transform.iloc[frame].values
    calc_transform_matrix = calc_transform.iloc[frame].values

    astr_real_vertices = apply_transformation(astr_vertices, astr_transform_matrix)
    calc_real_vertices = apply_transformation(calc_vertices, calc_transform_matrix)

    # minimum distance between facets
    pair_distances = cdist(astr_real_vertices, calc_real_vertices)
    min_distance = np.min(pair_distances)

    # centroid distance between facets
    astr_center = np.mean(astr_real_vertices, axis=0)
    calc_center = np.mean(calc_real_vertices, axis=0)
    center_distance = np.linalg.norm(astr_center - calc_center)

    # contact percentage
    threshold = 0.5
    all_min_distances = np.min(pair_distances, axis=1)
    astr_indices_in_contact = np.where(all_min_distances <= threshold)[0]
    astr_vertices_in_contact = len(astr_indices_in_contact)
    astr_contact_pct = (astr_vertices_in_contact / len(astr_real_vertices)) * 100

    # reduce vertices to 50 to represent surface, then turn minimum distances into heatmap

    # center of contact
    # if astr_vertices_in_contact > 0:
    #     coc = np.mean(astr_real_vertices[astr_indices_in_contact], axis=0)
    # else:
    #     coc = (np.nan, np.nan, np.nan)

    distances.append({'frame': frame + 1, 'min_distance': min_distance, 'center_distance': center_distance, 'contact_percentage' : astr_contact_pct})

distances_dataframe = pd.DataFrame(distances)
# scaling_factor = 3.5 / 52.0
# distances_dataframe['min_distance'] = distances_dataframe['min_distance'] * scaling_factor
# distances_dataframe['center_distance'] = distances_dataframe['center_distance'] * scaling_factor
print(distances_dataframe)

plt.figure(figsize=(10, 6))
plt.plot(distances_dataframe['frame'], distances_dataframe['min_distance'], label='Min Surface Distance', color='blue', linewidth=2)
plt.plot(distances_dataframe['frame'], distances_dataframe['center_distance'], label='Center Distance', color='red', linewidth=2)
plt.title('Subtalar Joint Space Over Time', fontsize=14)
plt.xlabel('Frame', fontsize=12)
plt.ylabel('Distance (mm)', fontsize=12)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.fill_between(distances_dataframe['frame'], distances_dataframe['contact_percentage'], color='blue', alpha=0.2)
plt.plot(distances_dataframe['frame'], distances_dataframe['contact_percentage'], label='Contact Area %', color='blue', linewidth=2)
plt.title(f'Subtalar Joint Contact Area Over Time with Threshold of {threshold} cm', fontsize=14)
plt.xlabel('Frame', fontsize=12)
plt.ylabel('Contact Area (%)', fontsize=12)
plt.legend()
plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(distances_dataframe['frame'], distances_dataframe['center_of_contact_x'], label='Coc X-axis', color='blue', linewidth=2)
# plt.plot(distances_dataframe['frame'], distances_dataframe['center_of_contact_y'], label='Coc Y-axis', color='red', linewidth=2)
# plt.plot(distances_dataframe['frame'], distances_dataframe['center_of_contact_z'], label='Coc Z-axis', color='yellow', linewidth=2)
# plt.title('Subtalar Joint Center of Contact (Coc) Over Time', fontsize=14)
# plt.xlabel('Frame', fontsize=12)
# plt.ylabel('Position (mm)', fontsize=12)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()