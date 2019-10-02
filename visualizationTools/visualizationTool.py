#!/usr/bin/env python3

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import glob
import argparse as ap

parser = ap.ArgumentParser(description='Visualization of Landau-de Gennes relaxation results',
													 epilog=
													'Note: \"infileprefix\" is a required argument with no flags. *******************   \
													 Minimal example usage: vis.py myfileprefix *********************************** \
													 Another example usage: vis.py myfileprefix -x 100 -y 50 150 -s 10 -ys 5 20 -d 0.4 -o 0 *********************************************************************')
parser.add_argument('infileprefix',# dest = 'infileprefix', default = 'data/test',
										help='(REQUIRED, with no - flag) file name of Q-tensor data to be imported, including path but excluding suffix \'_x#y#z#.txt\'')
parser.add_argument('-x', '--xslices', dest = 'xslices', nargs = '+', type = int, default = [],
										help='x-values of director field slices normal to x')
parser.add_argument('-y', '--yslices', dest = 'yslices', nargs = '+', type = int, default = [],
										help='y-values of director field slices normal to y')
parser.add_argument('-z', '--zslices', dest = 'zslices', nargs = '+', type = int, default = [],
										help='z-values of director field slices normal to z')
parser.add_argument('-s', '--stride', dest='stride', type = int, default = 1,
										help='number of sites between neighboring plotted points in director field slices')
parser.add_argument('-xs', '--xstride', dest='xstride', nargs = '+', type = int, default = [],
										help='number of sites between neighboring plotted points in each director field x slice. Multiple values may be given for multiple slices.')
parser.add_argument('-ys', '--ystride', dest='ystride', nargs = '+', type = int, default = [],
										help='number of sites between neighboring plotted points in each director field y slice. Multiple values may be given for multiple slices.')
parser.add_argument('-zs', '--zstride', dest='zstride', nargs = '+', type = int, default = [],
										help='number of sites between neighboring plotted points in each director field z slice. Multiple values may be given for multiple slices.')
parser.add_argument('-d', '--dthresh', dest='dthresh', type = float, default = 0.5*0.53,
										help='threshold for leading eigenvalue S below which a point is labeled a defect (default: %(default)s)')
parser.add_argument('-o', '--objects', dest='show_objects_int', type = int, default = 1,
										help='show (1) or don\'t show (0) boundary objects')


args = parser.parse_args() # get arguments from command line
infileprefix = args.infileprefix
stride = args.stride
xstride = args.xstride
ystride = args.ystride
zstride = args.zstride
dthresh = args.dthresh
show_objects = bool(args.show_objects_int)
# use last given value of xstride, ystride, or zstride to fill in any values given individually
# if none is given, use stride
if len(xstride) == 0:
	xstride = [stride]
if len(ystride) == 0:
	ystride = [stride]
if len(zstride) == 0:
	zstride = [stride]
last_given_xstride = xstride[-1]
for i in range(len(xstride), len(args.xslices)):
	xstride.append(last_given_xstride)
last_given_ystride = ystride[-1]
for i in range(len(ystride), len(args.yslices)):
	ystride.append(last_given_ystride)
last_given_zstride = zstride[-1]
for i in range(len(zstride), len(args.zslices)):
	zstride.append(last_given_zstride)
# arrange all director slices into a single list of [ index (0=x,1=y,2=z), slice_value, stride ]
nslices_info = [ [ 0, args.xslices[i], xstride[i] ] for i in range(len(args.xslices)) ] + \
							[ [ 1, args.yslices[i], ystride[i] ] for i in range(len(args.yslices)) ] + \
							[ [ 2, args.zslices[i], zstride[i] ] for i in range(len(args.zslices)) ]

# infileprefix = 'data/test12'# files of Q-tensor data to be imported, before '_x#y#z#.txt'
# dthresh = 0.5*0.53 # threshold for S below which a point is a defect
# stride = 10 # skip this many sites between neighboring plotted directors

def Qfromq(q):
	""" converts five-element set q of unique Q-tensor elements to the full 3x3 Q-tensor matrix """
	return np.array( [ [ q[0], q[1], q[2] ],
										[ q[1], q[3], q[4] ],
										[ q[2], q[4], -q[0] - q[3] ] ] )

def nfromQ(Q):
	""" extracts director from Q-tensor as eigenvector of leading eigenvalue """
	evals, evecs = LA.eig(Q)
	return evecs[ np.argmax(evals) ] # eigenvector of leading eigenvalue




# read in files
infilenames = glob.glob(infileprefix+'_x*y*z*.txt')
infilenames.sort()
if len(infilenames) == 0:
	print('Error: no files found of the form ' + infileprefix+'_x*y*z*.txt')
	exit()

print('Loading these files:')
for infilename in infilenames:
	print( '\t'+infilename )
dat = np.concatenate( np.array( [ np.loadtxt(infilename) for infilename in infilenames ] ) )
print('Finished loading files')

print('Finding defect points')
x,y,z,qxx,qxy,qxz,qyy,qyz,sitetype,S = dat.T
# defects: select liquid crystal sites with S below dthresh
dpts = dat[ (S < dthresh) * (sitetype <= 0) ] # multiplication of boolean arrays becomes logical AND # \

# objects: select sites not part of the liquid crystal
if show_objects:
	print('Finding boundary object points')
	objpts = dat[ sitetype > 0 ] 
else:
	objpts = np.empty([0,3])

if len(nslices_info) > 0:
	print('Creating director field slices')
# x,y,z bounds
lims = np.array( [ np.min(dat[:,0:3],axis=0) , np.max(dat[:,0:3],axis=0) ], dtype=int )
[ [ xmin, ymin, zmin ], [ xmax, ymax, zmax ] ] = lims

# get second-smallest values in each direction to get stride of imported data
data_strides = [ dat[ np.argmax(dat[:,i] > (dat[:,i])[0]) ][i] for i in range(3) ]
data_dims = [ int( int(lims[1,i] - lims[0,i] ) / data_strides[i] ) + 1 for i in range(3) ]

# store data in a matrix indices x,y,z
mat = np.transpose(dat.reshape(data_dims[2],data_dims[1],data_dims[0],(dat.shape)[-1]),(2,1,0,3))



# mesh of (x,y,z) for director slices
nslice_data = []
for nslice_info in nslices_info:
	slice_norm_index, slice_norm_val, stride = nslice_info
	data_stride = data_strides[slice_norm_index]
	slice_norm_val = data_stride * int(slice_norm_val / data_stride)
	stride = data_stride * max( 1, int(stride / data_stride) )
	if slice_norm_index == 0:
		npts_x, npts_y, npts_z = np.meshgrid(np.arange(slice_norm_val,slice_norm_val+1,2, dtype = int),
																			 np.arange(ymin,ymax+1,stride, dtype = int),
																			 np.arange(zmin,zmax+1,stride, dtype = int))
	elif slice_norm_index == 1:
		npts_x, npts_y, npts_z = np.meshgrid(np.arange(xmin,xmax+1,stride, dtype = int),
																			 np.arange(slice_norm_val,slice_norm_val+1,2, dtype = int),
																			 np.arange(zmin,zmax+1,stride, dtype = int))
	elif slice_norm_index == 2:
		npts_x, npts_y, npts_z = np.meshgrid(np.arange(xmin,xmax+1,stride, dtype = int),
																			 np.arange(ymin,ymax+1,stride, dtype = int),
																			 np.arange(slice_norm_val,slice_norm_val+1,2, dtype = int))
	npts_shape = npts_x.shape

	# get director from Q as a matrix with indices x,y,z
	n_vals = np.empty([npts_shape[0],npts_shape[1],npts_shape[2],3])
	for i in range(npts_shape[0]):
		for j in range(npts_shape[1]):
			for k in range(npts_shape[2]):
				q = mat[int(npts_x[i,j,k]/data_strides[0]),int(npts_y[i,j,k]/data_strides[1]), \
					 int(npts_z[i,j,k]/data_strides[2])]
				Q = Qfromq(q)
				n_vals[i,j,k] = nfromQ(Q)
	nslice_data.append([npts_x,npts_y,npts_z,n_vals])

print('Making plot')
fig = plt.figure(figsize=(10,10)) # specify size of figure in inches
ax = fig.add_subplot(111, projection='3d')


# plot the objects
if objpts.shape[0] > 0:
	ax.scatter(objpts[:,0], objpts[:,1], objpts[:,2], c='orange', alpha=0.5, zorder=-1, s=0.005)

#plot the director and its opposite to make arrows double-headed
for nslice in nslice_data:
	npts_x, npts_y, npts_z, n_vals = nslice
	for flipfactor in [-1,1]:
		ax.quiver(npts_x, npts_y, npts_z,
					flipfactor * n_vals[:,:,:,0], flipfactor * n_vals[:,:,:,1], flipfactor * n_vals[:,:,:,2],
					length=stride,normalize=False,pivot='middle',color='purple')
# plot the defects
if dpts.shape[0] > 0:
	ax.scatter(dpts[:,0], dpts[:,1], dpts[:,2], c='green')


# make the plot range a cube (i.e. equal side lengths)
max_dim = max(lims[1]-lims[0])
xmax_plot_range = max(xmax, xmin + max_dim)
ymax_plot_range = max(ymax, ymin + max_dim)
zmax_plot_range = max(zmax, zmin + max_dim)
ax.set_xlim3d(xmin,xmax_plot_range)
ax.set_ylim3d(ymin,ymax_plot_range)
ax.set_zlim3d(zmin,zmax_plot_range)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.title((infileprefix.split('/'))[-1]) # make plot title the imported file name, without the file type and the parent folder names

plt.show()
