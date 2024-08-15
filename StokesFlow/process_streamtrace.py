#!/usr/bin/env python

import os
import sys
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import skimage as sk
from skimage import measure, io
from rdp import rdp
import numpy as np
from PIL import Image


def save_inner_shape(shape_grid, z_y_seeds, image_filename, nx, ny):

    job_dir = os.path.dirname(image_filename)
    if job_dir == '':
        job_dir = '.'

    z_y_seeds = np.array(z_y_seeds)
    np.savetxt('{}/inner_contour_seeds.csv'.format(job_dir), z_y_seeds,
            header='x, y, z,', delimiter=',', fmt='%.9f')


    img_array = shape_grid
    w,h = img_array.shape

    img_array = img_array.astype(np.uint8)

    zero_loc = np.where(img_array == 0)
    color_loc = np.where(img_array == img_array.max())

    img_array[zero_loc] = 255
    img_array[color_loc] = 0.0

    img_r = img_array.copy()
    img_g = img_array.copy()
    img_b = img_array.copy()

    img_r[color_loc] = 81
    img_g[color_loc] = 164
    img_b[color_loc] = 209

    img_out = np.zeros((w,h,3), dtype=np.uint8)
    img_out[:,:,0] = img_r
    img_out[:,:,1] = img_g
    img_out[:,:,2] = img_b

    img = Image.fromarray(img_out,'RGB')

    img = img.rotate(90)
    img.save(image_filename)

def create_inner_contour(contour_filename, nx, ny, span):
    contour_points = read_contour_file(contour_filename)

    [grid, z_y_seeds ] = get_inner_shape(contour_points, nx, ny, span)

    return [grid, z_y_seeds]

def get_inner_shape(contour_points, nx, ny, span):
    # Flip contour points (why? don't know, images are fun)
    for n in range(len(contour_points)):
        contour_points[n][0] *= -1.0
    polygon = Polygon(contour_points)

    y_min = -0.5*span
    y_max = 0.5*span
    x_min = -0.5*span
    x_max = 0.5*span

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)

    z_y_seeds = []

    grid = np.zeros((nx,ny), dtype=np.uint8)

    # Get high-res inlet profile shape
    for j in range(0, ny):
        for i in range(0, nx):
            point = Point(x[i], y[j])

            if polygon.contains(point):
                grid[i,j] = 255

    # Get low-res streamtrace seeds for visualization
    ny_low = 200
    nx_low = ny_low
    x_low = np.linspace(x_min, x_max, nx_low)
    y_low = np.linspace(y_min, y_max, ny_low)

    for j in range(0, ny_low):
        for i in range(0, nx_low):
            point = Point(x_low[i], y_low[j])
            if polygon.contains(point):
                z_y_seeds.append([0.0, -x_low[i], y_low[j]])

    return [grid, z_y_seeds]

def process_inner_shape(contour_filename, image_filename, span=1.0):
    # Hardcoded inner flow shape image at 400x400  to match typical
    # streamtrace resolution, super necessary for anything else
    nx = 400
    ny = 400

    [grid, z_y_seeds ] = create_inner_contour(contour_filename, nx, ny, span)

    save_inner_shape(grid, z_y_seeds, image_filename, nx, ny)

def advection_to_index(adata, nx, ny, span=1.0):
    if (len(np.shape(adata)) == 1):
        # Using adata directly
        index_map = adata_to_index(adata, nx, ny, span)
        return index_map
    elif (np.shape(adata)[1] == 4):
        # Doing quiver to adata conversion
        adata = quiver_to_adata(adata, nx, ny, span)
        index_map = adata_to_index(adata, nx, ny, span)
        return index_map
    else:
        sys.exit('Unrecognized advection map format. Exiting')

def check_proximity(target, value):
    eps = 1e-5

    if np.abs(target-value) < eps:
        return True
    else:
        return False

def quiver_to_adata(qdata_raw, nx, ny, span=1.0):

    print('Doing quiver to index')

    qdata = np.zeros((nx*ny, 4))

    x_list = np.round(np.linspace(-0.5*span, 0.5*span, nx), 6)
    y_list = np.round(np.linspace(-0.5*span, 0.5*span, ny), 6)

    for n in range(len(qdata_raw)):
        counter = 0

        x = qdata_raw[n,0]
        y = qdata_raw[n,1]

        print('n = {}'.format(n))

        for j in range(ny):
            for i in range(nx):
                if (check_proximity(x, x_list[i]) and check_proximity(y, y_list[j])):
                    qdata[counter][0] = x
                    qdata[counter][1] = y
                    qdata[counter][2] = qdata_raw[n][2]
                    qdata[counter][3] = qdata_raw[n][3]

                counter += 1

    adata = np.reshape(qdata[:,3:4], (ny*nz*2, 1), order='F')

    print('Adata from qdata:')
    print(adata.shape)

    return adata

def adata_to_index(adata, nx, ny,span):
    # Convert advection map to lookup table ("index map")

    scaled_dx = float(nx)/float(span)
    scaled_dy = float(ny)/float(span)

    origins = range(nx*ny)
    destinations = np.zeros(nx*ny, dtype=int)

    for cell in origins:
        x_o = int(cell % nx)
        y_o = int(cell / nx)

        dx = round(adata[cell] * scaled_dx)
        dy = round(adata[cell + nx*ny] * scaled_dx)

        x_d = int(x_o + dx)
        y_d = int(y_o + dy)

        if (x_d < 0):
            x_d = 0
        if (y_d < 0):
            y_d = 0
        if (x_d > nx-1):
            x_d = nx-1
        if (y_d > ny-1):
            y_d = ny-1

        destinations[cell] = int((y_d * nx) + x_d)

    return destinations

def transform_flow_image(sequence, outlet, perm_maps, nx, ny):
    '''
    :param sequence: pillar sequence as list of integers
    :param inlet: inlet vector created from form_inlet
    :param perm_data: list of permutation vectors
    :param nx: streamtrace resolution in width dimension (y direction in 3D flow)
    :param ny: streamtrace resolution in height dimesion (z direction in 3D flow)
    :return: binary sculpted flow image
    '''
    # Use permutation matrices with inlet vector to form fluid deformation image

    if (len(sequence) == 1):
        outlet = outlet[perm_maps[sequence[0]]]
    else:
        for n in range(0, len(sequence)):
            outlet = outlet[perm_maps[sequence[n]]]

    outlet = np.rot90(outlet.reshape(nx,ny, order='F'))

    return outlet

def create_outlet_flow_shape(st_filename, nx ,ny, inlet, span=1.0):
    # Load advection map
    try:
        amap = np.loadtxt(st_filename)
    except:
        print('Could not load {}'.format(st_filename))

    # Create index map for forward model
    '''
    if (len(np.shape(amap))>1 and np.shape(amap)[1] == 4):
        #try:
        print('Doing conversion')
        convert_cmd = '{}/convert_qdata_to_adata {} {} {} {}'.format(
                FS_DIR, st_filename, nx, ny, span)
        subprocess.run(convert_cmd, stdout=subprocess.PIPE, shell=True)

        amap = np.loadtxt('adata_converted.txt')
        amap *= -1.0
        #except:
        #    print('Could not convert quiver to adata')

    '''
    #try:
    index_map = advection_to_index(amap, nx, ny, span)
    #except:
    #    print('Could not convert advection to index map')

    index_maps = []
    index_maps.append(index_map)

    # Create flow image
    try:
        flow_img = transform_flow_image([0], inlet, index_maps, nx, ny)
    except:
        print('Could not compute flow image using fs.flow_image')

    # Need to flip lr for some reason
    flow_img = np.fliplr(flow_img)

    return flow_img

def save_outlet_flow_shape(img_array, outlet_filename):
    # Create actual image

    w,h = img_array.shape

    img_array = img_array.astype(np.uint8)

    zero_loc = np.where(img_array == 0)
    color_loc = np.where(img_array == img_array.max())

    img_array[zero_loc] = 255
    img_array[color_loc] = 0.0

    img_r = img_array.copy()
    img_g = img_array.copy()
    img_b = img_array.copy()

    img_r[color_loc] = 81
    img_g[color_loc] = 164
    img_b[color_loc] = 209

    img_out = np.zeros((w,h,3), dtype=np.uint8)
    img_out[:,:,0] = img_r
    img_out[:,:,1] = img_g
    img_out[:,:,2] = img_b

    img = Image.fromarray(img_out,'RGB')
    #img = Image.fromarray(flow_img)
    #img = img.convert("L")
    img.save(outlet_filename)

def read_contour_file(filename):
    contour_points = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            points = list(map(float, line.split(',')))
            contour_points.append(points)
            line = f.readline()
    return contour_points

def get_inner_shape_as_inlet_from_file(filename, inlet_nx, inlet_ny, span=1.0):
    contour_points = read_contour_file(filename)

    inlet = get_inner_shape_as_inlet(contour_points, inlet_nx, inlet_ny, span)

    return inlet

def get_inner_shape_as_inlet(contour_points, inlet_nx, inlet_ny, span=1.0):

    [grid, y_z_seeds] = get_inner_shape(contour_points, inlet_nx, inlet_ny, span)

    grid = np.rot90(grid,3)

    inlet = grid.reshape((1, inlet_nx*inlet_ny)).squeeze()

    return inlet

def get_contours(gray_img):
    height, width = gray_img.shape
    # Normalize and flip (for some reason)
    contours = sk.measure.find_contours(gray_img, 0.5)

    print('Found {} contours'.format(len(contours)))
    for n, contour in enumerate(contours):

        contour[:,1] -= 0.5 * height
        contour[:,1] /= height

        contour[:,0] -= 0.5 * width
        contour[:,0] /= width
        contour[:,0] *= -1.0

    return contours

def optimize_contour(contour):
    print("Optimizing contour.")
    ## Use low-pass fft to smooth out
    x = contour[:,1]
    y = contour[:,0]

    signal = x + 1j*y

    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.shape[-1])
    cutoff = 0.12
    fft[np.abs(freq) > cutoff] = 0

    signal_filt = np.fft.ifft(fft)

    contour[:,1] = signal_filt.real
    contour[:,0] = signal_filt.imag

    contour = rdp(contour, epsilon=0.0005)

    # Remove final point in RDP, which coincides with
    # the first point
    contour = np.delete(contour, len(contour)-1, 0)

    # Figure out a reasonable radius
    max_x = max(contour[:,1])
    min_x = min(contour[:,1])

    max_y = max(contour[:,0])
    min_y = min(contour[:,0])

    # Set characteristic lengths, epsilon cutoff
    lc = min((max_x - min_x), (max_y - min_y))
    mesh_lc = 0.01 * lc

    return [contour, mesh_lc]

def load_image(img_fname):

    print('Loading image {}'.format(img_fname))
    img = sk.io.imread(img_fname)

    # print(img.shape)
    if (len(img.shape) == 2):
        gray_img = img
    else:
        if (img.shape[2] == 3):
            gray_img = sk.color.rgb2gray(img)
        if (img.shape[2] == 4):
            rgb_img = sk.color.rgba2rgb(img)
            gray_img = sk.color.rgb2gray(rgb_img)

    return gray_img

def write_contour_to_file(contour, fname):
    with open(fname, 'w') as f:
        for point in contour:
            f.write('{:.6e}, {:.6e}\n'.format(point[1], point[0]))

def process_st_result(img_fname, st_fname, nx, ny):
    # Determine inner flow shape at inlet
    img = load_image(img_fname)
    contours = get_contours(img)
    [inner_contour, mesh_lc] = optimize_contour(contours[1])

    # Save contour points 
    inner_contour_fname = 'inner_contour.txt'
    write_contour_to_file(inner_contour, inner_contour_fname)

    print('Getting inner contour.')
    inlet = get_inner_shape_as_inlet(inner_contour, nx, ny)

    print('Processing flow shape.')
    flow_img = create_outlet_flow_shape(st_fname, nx, ny, inlet)
    outlet_filename = 'outlet_shape.png'
    save_outlet_flow_shape(flow_img, outlet_filename)

    print('Saving inner shape')
    image_filename = 'inner_shape.png'
    inner_shape_filename = 'inner_contour.txt'
    process_inner_shape(inner_shape_filename, image_filename)

if __name__ == '__main__':
    # Inlet geometry image
    img_fname = sys.argv[1]

    # Reverse streamtrace file
    st_fname = sys.argv[2]

    # Get streamtrace resolution
    if len(sys.argv) > 3:
        nx = int(sys.argv[3])
        ny = int(sys.argv[3])
    else:
        # If no resolution provided, assume 400x400
        nx = 400
        ny = 400

    process_st_result(img_fname, st_fname, nx, ny)


