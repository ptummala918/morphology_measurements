import numpy as np
import pyvista as pv
import os

CONTACT_THRESHOLD = 0.05   # cm
MAX_DIST          = 0.2    # cm 
FPS               = 25
OUTPUT_PATH       = 'results/facet_heatmap.mp4'

os.makedirs('results', exist_ok=True)

# loads objects and convert to polydata form
def load_obj(path, scale=0.1):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('f '):
                idx = [int(tok.split('/')[0]) - 1 for tok in line.split()[1:]]
                faces.append(idx)
    verts = np.array(verts) * scale
    faces_pv = np.array([[3] + f for f in faces]).flatten()
    return pv.PolyData(verts, faces_pv)

# loads meshes
astr_mesh = load_obj('data/astr_EF.obj')
calc_mesh = load_obj('data/calc_EF.obj')

# load the vertex distances from measure.py
astr_dists = np.load('data/astr_vertex_distances.npy') 
calc_dists = np.load('data/calc_vertex_distances.npy')
n_frames   = astr_dists.shape[0]

# places meshes side by side for visualization purposes (may change to show real movement)
astr_mesh.points -= astr_mesh.points.mean(axis=0)
calc_mesh.points -= calc_mesh.points.mean(axis=0)
gap    = 0.05
extent = astr_mesh.bounds[1] - astr_mesh.bounds[0]      
calc_mesh.points[:, 0] += extent + gap

# sets key to use by plotter
astr_mesh.point_data['distance'] = astr_dists[0].astype(np.float32)
calc_mesh.point_data['distance'] = calc_dists[0].astype(np.float32)

clim = [0.0, MAX_DIST]

# builds plotter and actors
plotter = pv.Plotter(off_screen=True, window_size=[1280, 720])
plotter.set_background('#111111')

astr_actor = plotter.add_mesh(
    astr_mesh, scalars='distance', clim=clim, cmap='jet_r',
    show_scalar_bar=False, smooth_shading=True,
)
calc_actor = plotter.add_mesh(
    calc_mesh, scalars='distance', clim=clim, cmap='jet_r',
    show_scalar_bar=False, smooth_shading=True,
)

# scalar bar bound to astr_actor
plotter.add_scalar_bar(
    title='Min Distance (cm)', n_labels=5, fmt='%.2f', color='white',
    title_font_size=14, label_font_size=12,
    position_x=0.85, position_y=0.25, width=0.08, height=0.5,
    mapper=astr_actor.GetMapper(),
)

plotter.add_text('Astragalus\nEctal Facet', position=(0.18, 0.05),
                 font_size=11, color='white', viewport=True)
plotter.add_text('Calcaneus\nEctal Facet',  position=(0.55, 0.05),
                 font_size=11, color='white', viewport=True)
frame_text = plotter.add_text(f'Frame 1 / {n_frames}', position=(0.02, 0.95),
                               font_size=12, color='#44aaff', viewport=True)

# camera configuration
plotter.view_isometric()
plotter.camera.zoom(1.3)

print(f"Rendering {n_frames} frames → {OUTPUT_PATH}")
plotter.open_movie(OUTPUT_PATH, framerate=FPS)

for i in range(n_frames):
    astr_mesh.point_data['distance'] = astr_dists[i].astype(np.float32)
    calc_mesh.point_data['distance'] = calc_dists[i].astype(np.float32)

    frame_text.SetInput(f'Frame {i+1} / {n_frames}')
    plotter.render()
    plotter.write_frame()

    if (i + 1) % 100 == 0:
        print(f'  {i+1}/{n_frames}')

plotter.close()
print(f'Done — saved to {OUTPUT_PATH}')