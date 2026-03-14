import numpy as np
import pyvista as pv

CONTACT_THRESHOLD = 0.05   # cm — used for scalar bar annotation only
MAX_DIST          = 0.2   # cm — clips the color scale (anything >= this is "far")
FPS               = 25
OUTPUT_PATH       = 'results/facet_heatmap.mp4'

# to load the object files as pyVista polydata files
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

# load the meshes and convert them to polydata files
astr_mesh = load_obj('data/astr_EF.obj')
calc_mesh = load_obj('data/calc_EF.obj')

# load the distance calculations from measure.py
astr_dists = np.load('data/astr_vertex_distances.npy')
calc_dists = np.load('data/calc_vertex_distances.npy')
n_frames   = astr_dists.shape[0]

# this places meshes next to each other
astr_mesh.points -= astr_mesh.points.mean(axis=0)
calc_mesh.points -= calc_mesh.points.mean(axis=0)
gap = 0.05
extent = astr_mesh.bounds[1] - astr_mesh.bounds[0]
calc_mesh.points[:, 0] += extent + gap

# creates plot
plotter = pv.Plotter(off_screen=True, window_size=[1280, 720])
plotter.set_background('#111111')
clim = [0.0, MAX_DIST]

astr_actor = plotter.add_mesh(
    astr_mesh, scalars=astr_dists[0], clim=clim, cmap='jet_r',
    show_scalar_bar=False, smooth_shading=True,
)
calc_actor = plotter.add_mesh(
    calc_mesh, scalars=calc_dists[0], clim=clim, cmap='jet_r',
    show_scalar_bar=False, smooth_shading=True,
)

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
frame_text = plotter.add_text('Frame 1 / 650', position=(0.02, 0.95),
                               font_size=12, color='#44aaff', viewport=True)

plotter.view_isometric()
plotter.camera.zoom(1.3)

# renders animation
print(f"Rendering {n_frames} frames to {OUTPUT_PATH} ...")
plotter.open_movie(OUTPUT_PATH, framerate=FPS)

for i in range(n_frames):
    astr_mesh.point_data['scalars'] = astr_dists[i]
    calc_mesh.point_data['scalars'] = calc_dists[i]
    frame_text.SetInput(f'Frame {i+1} / {n_frames}')
    plotter.render()
    plotter.write_frame()
    if (i + 1) % 100 == 0:
        print(f'  {i+1}/{n_frames}')

plotter.close()
print(f'Done — saved to {OUTPUT_PATH}')