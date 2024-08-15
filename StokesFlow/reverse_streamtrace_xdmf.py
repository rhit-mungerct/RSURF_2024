# NEEDS TO BE RUN THROUGH PARAVIEW'S "pvpython" 
from paraview.simple import *
import vtk.numpy_interface.dataset_adapter as dsa
import numpy as np

import sys

def generate_seeds(x_min, x_max, y_min, y_max, z_min, z_max, Ny, Nz):
    y = np.linspace(y_min, y_max, Ny)
    z = np.linspace(z_min, z_max, Nz)

    z_y_seeds = np.zeros((Ny*Nz, 3))

    seed_counter = 0

    for k in range(0, Nz):
        for j in range(0, Ny):
            z_y_seeds[seed_counter] = (x_max, y[j], z[k])
            seed_counter += 1

    csv_data_fname = './reverse_ST_seeds_Ny_{}_Nz_{}.csv'.format(Ny,Nz)
    np.savetxt(csv_data_fname, z_y_seeds, header='x, y, z', delimiter=",", fmt='%.9f')


def print_point_data(point):
    print("x={:.3f}, y={:.3f}, z={:.3f}\n".format(point[0], point[1], point[2]))


def check_separation(point_data):
    delta_x = point_data[1:,0] - point_data[:-1,0]
    # print(delta_x)
    if any(x>0.0 for x in delta_x):
        #print(delta_x)
        return True
    return False


data3D_fname = sys.argv[1]
Ny = int(sys.argv[2])
Nz = int(sys.argv[3])
advection_fname = sys.argv[4]

datan_dat = Xdmf3ReaderS(registrationName=data3D_fname, FileName=[data3D_fname])

Show(datan_dat)
#datan_dat.UpdatePipeline()

(x_min, x_max, y_min, y_max, z_min, z_max) = datan_dat.GetDataInformation().GetBounds()

print("x_min = {:.3f}, x_max = {:.3f}".format(x_min, x_max))
print("y_min = {:.3f}, y_max = {:.3f}".format(y_min, y_max))
print("z_min = {:.3f}, z_max = {:.3f}".format(z_min, z_max))

y = np.linspace(y_min, y_max, Ny)
z = np.linspace(z_min, z_max, Nz)

# Read in streamtrace seed points
generate_seeds(x_min, x_max, y_min, y_max, z_min, z_max, Ny, Nz)
foocsv_fname = './reverse_ST_seeds_Ny_%d_Nz_%d.csv'%(Ny,Nz)
foocsv = CSVReader(registrationName=foocsv_fname, FileName=[foocsv_fname])

# Create a new 'Table To Points'
tableToPoints1 = TableToPoints(Input=foocsv)
tableToPoints1.XColumn = '# x'
tableToPoints1.YColumn = ' y'
tableToPoints1.ZColumn = ' z'

# set active source
SetActiveSource(datan_dat)

# create a new 'Stream Tracer With Custom Source'
custom_tracer1 = StreamTracerWithCustomSource(Input=datan_dat,
    SeedSource=tableToPoints1)
custom_tracer1.SurfaceStreamlines = 0
custom_tracer1.IntegrationDirection = 'BACKWARD'
custom_tracer1.IntegratorType = 'Runge-Kutta 4-5'
custom_tracer1.IntegrationStepUnit = 'Cell Length'
custom_tracer1.InitialStepLength = 0.1
custom_tracer1.MinimumStepLength = 0.02
custom_tracer1.MaximumStepLength = 0.5
custom_tracer1.MaximumSteps = int(5000 * (x_max - x_min))
custom_tracer1.MaximumStreamlineLength = 2 * (x_max - x_min)
custom_tracer1.TerminalSpeed = 1e-12
custom_tracer1.MaximumError = 1e-06

vtkdata = servermanager.Fetch(custom_tracer1)
data = dsa.WrapDataObject(vtkdata)

point_data = data.Points

print(point_data)

# Post-processing
x_start_locations = np.where(point_data[:,0] == x_max)[0]

#print(x_start_locations)

dy_dz = np.zeros((Ny*Nz, 2))

counter = 0

print('num x_start_locations = {}'.format(len(x_start_locations)))

for i in range(len(x_start_locations)):
    # See if it's the last tracer
    if i == len(x_start_locations)-1:
        x_end_data = point_data[-1]
        # See if last tracer goes anywhere
        if x_end_data[0] == point_data[x_start_locations[i]][0]:
            # print("Last tracer doesn't go anywhere!")
            dy_dz[counter] = (0.0, 0.0)
            counter += 1
            break
        # Otherwise, record the last movement
        advection = point_data[x_start_locations[i]] - x_end_data
        dy_dz[counter] = advection[1:]
        break
    # See if the tracer even goes anywhere
    if x_start_locations[i+1] == x_start_locations[i]+1:
        dy_dz[counter] = (0.0, 0.0)
        counter += 1
        continue
    x_end_location = x_start_locations[i+1] - 1
    # See if the tracer ended prematurely
    if point_data[x_end_location][0] > x_min+0.1:
        dy_dz[counter] = (0.0, 0.0)
        counter += 1
        continue
    #if check_separation(point_data[x_start_locations[i]:x_end_location]):
    #    dy_dz[counter] = (0.0, 0.0)
    #    counter += 1
    #    continue
    advection = point_data[x_end_location] - point_data[x_start_locations[i]]

    dy_dz[counter] = advection[1:]
    counter += 1

# Reshape data
dy_dz_r = np.reshape(dy_dz, (Ny*Nz*2, 1), order='F')

np.savetxt(advection_fname, dy_dz_r, fmt='%.9f')

