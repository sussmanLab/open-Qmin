#!/usr/bin/env python3

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg as LA
import glob
import argparse as ap

dpt_size = 10 # size of point markers if scatter plot for defects
obj_size = 4 # size of point markers if scatter plot for objects
obj_stride = 1 # how many object points to skip plotting between every show point

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
parser.add_argument('-sf', '--savefig', dest='savefig_name', type = str, default = '',
										help='filename to save figure; leave blank to not save')
parser.add_argument('-sh', '--show', dest='b_show_fig', type = int, default = 1,
										help='show (1) or don\'t show (0) figure interactively')
parser.add_argument('-el', '--elev', dest='view_elev', type = float, default = 10,
										help='initial view elevation')
parser.add_argument('-az', '--azim', dest='view_azim', type = float, default = 135,
										help='initial view azimuthal angle')

args = parser.parse_args() # get arguments from command line
infileprefix = args.infileprefix
stride = args.stride
xstride = args.xstride
ystride = args.ystride
zstride = args.zstride
dthresh = args.dthresh
show_objects = bool(args.show_objects_int)
savefig_name = args.savefig_name
b_save_fig = ( savefig_name != '' ) # boolean, true only if a savefig_name was given
print(savefig_name, b_save_fig)
b_show_fig = bool(args.b_show_fig)
view_elev = args.view_elev
view_azim = args.view_azim
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

def Qfromq(q):
	""" converts five-element set q of unique Q-tensor elements to the full 3x3 Q-tensor matrix """
	return np.array( [ [ q[0], q[1], q[2] ],
										[ q[1], q[3], q[4] ],
										[ q[2], q[4], -q[0] - q[3] ] ] )

def nfromQ(Q):
	""" extracts director from Q-tensor as eigenvector of leading eigenvalue """
	evals, evecmat = LA.eigh(Q)
	return evecmat[ :, np.argmax(evals) ] # eigenvector of leading eigenvalue, as column of eigenvector matrix


# read in files
infilenames = glob.glob(infileprefix+'_x*y*z*.txt')
infilenames.sort()
if len(infilenames) == 0:
	print('Error: no files found of the form ' + infileprefix+'_x*y*z*.txt')
	exit()

print('Loading these files:')
for infilename in infilenames:
	print( '\t'+infilename )
dat = np.concatenate( np.array( [ pd.read_csv(infilename,sep='\t',header=None).to_numpy() for infilename in infilenames  ] ) ) # Note: must have recent version of package 'pandas' to use this line; otherwise use the next line
# dat = np.concatenate( np.array( [ pd.read_csv(infilename,sep='\t',header=None).values for infilename in infilenames  ] ) )
print('Finished loading files')

# x,y,z bounds
lims = np.array( [ np.min(dat[:,0:3],axis=0) , np.max(dat[:,0:3],axis=0) ], dtype=int )
[ [ xmin, ymin, zmin ], [ xmax, ymax, zmax ] ] = lims

# get second-smallest values in each direction to get stride of imported data
data_strides = [ dat[ np.argmax(dat[:,i] > (dat[:,i])[0]) ][i] for i in range(3) ]
data_dims = [ int( int(lims[1,i] - lims[0,i] ) / data_strides[i] ) + 1 for i in range(3) ]

print('Finding defect points')
x,y,z,qxx,qxy,qxz,qyy,qyz,sitetype,S = dat.T
# defects: select liquid crystal sites with S below dthresh
dpts = dat[ (S < dthresh) * (sitetype <= 0) ] # multiplication of boolean arrays becomes logical AND # \

# objects: select sites not part of the liquid crystal
if show_objects:
	print('Finding boundary object points')
	objpts = dat[ sitetype > 0 ][::obj_stride]
else:
	objpts = np.empty([0,3])

if len(nslices_info) > 0:
	print('Creating director field slices')

# store data in a matrix with indices x,y,z
#mat = np.transpose(dat.reshape(data_dims[2],data_dims[1],data_dims[0],(dat.shape)[-1]),(2,1,0,3))
mat = np.zeros((data_dims[0],data_dims[1],data_dims[2],(dat.shape)[-1]-3))
for i in range(len(dat)):
	line = dat[i]
	(x,y,z) = [ int((int(line[i]) - lims[0][i])/int(data_strides[i])) for i in range(3) ]
	mat[x,y,z] = line[3:]

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
				matline = mat[int(npts_x[i,j,k]/data_strides[0]),int(npts_y[i,j,k]/data_strides[1]), \
					 int(npts_z[i,j,k]/data_strides[2])]
				Q = Qfromq(matline[0:5])
				n_vals[i,j,k] = nfromQ(Q)
	nslice_data.append([npts_x,npts_y,npts_z,n_vals])

print('Making plot')
fig = plt.figure(figsize=(10,10)) # specify size of figure in inches
ax = fig.add_subplot(111, projection='3d')


# plot the objects
if objpts.shape[0] > 0:
	ax.scatter(objpts[:,0], objpts[:,1], objpts[:,2], c='yellow', alpha=0.5, zorder=-1, s=obj_size)

## to shift points through the periodic boundaries:
# shiftvec = [0,0,0]
shiftvec = [int((xmax-xmin)/4),int((ymax-ymin)/4),0]

## to rescale one or more of the axes
zstretch = 1
ystretch = 1
xstretch = 1

#plot the director and its opposite to make arrows double-headed
for nslice in nslice_data:
	npts_x, npts_y, npts_z, n_vals = nslice
	npts_x = xstretch * (xmin + (npts_x + shiftvec[0])%(xmax-xmin+1))
	npts_y = ystretch * (ymin + (npts_y + shiftvec[1])%(ymax-ymin+1))
	npts_z = zstretch * (zmin + (npts_z + shiftvec[2])%(zmax-zmin+1))
	for flipfactor in [-1,1]:
		ax.quiver(npts_x, npts_y, npts_z,
					flipfactor * n_vals[:,:,:,0], flipfactor * n_vals[:,:,:,1], flipfactor * n_vals[:,:,:,2],
					length=stride,normalize=False,pivot='middle',color='black',arrow_length_ratio=0,linewidth=0.5)
# plot the defects
if dpts.shape[0] > 0:
	dpts[:,0] = xstretch * (xmin + (dpts[:,0] + shiftvec[0])%(xmax-xmin+1))
	dpts[:,1] = ystretch * (ymin + (dpts[:,1] + shiftvec[1])%(ymax-ymin+1))
	dpts[:,2] = zstretch * (zmin + (dpts[:,2] + shiftvec[2])%(zmax-zmin+1))
	ax.scatter(dpts[:,0], dpts[:,1], dpts[:,2], c='blue', s=dpt_size)

""" Plot styling options """

fontname='DejaVu Sans'

# make the plot aspect ratio at most 4
max_dim = max(lims[1]-lims[0])
xmax_plot_range = max(xmax, xmin + 1*max_dim)
ymax_plot_range = max(ymax, ymin + 1*max_dim)
zmax_plot_range = max(zmax, zmin + 1*max_dim)

## for no axes ticks:
#ax.xaxis.set_ticks([])
#ax.yaxis.set_ticks([])
#ax.zaxis.set_ticks([])

## for three ticks along each axis:
tickfontsize = 12
xtick_values = range(0,xmax+1, int(xmax/2))
ax.xaxis.set_ticks( [ xstretch * elem for elem in xtick_values ] )
ax.xaxis.set_ticklabels(xtick_values,  fontsize=tickfontsize, fontname=fontname)
ytick_values = range(0,ymax+1, int(ymax/2))
ax.yaxis.set_ticks( [ ystretch * elem for elem in ytick_values ] )
ax.yaxis.set_ticklabels(ytick_values,  fontsize=tickfontsize, fontname=fontname)
ztick_values = range(0,zmax+1, int(zmax/2))
ax.zaxis.set_ticks( [ zstretch * elem for elem in ztick_values ] )
ax.zaxis.set_ticklabels(ztick_values, fontsize=tickfontsize, fontname=fontname)

ax.set_xlim3d(xmin,xmax_plot_range)
ax.set_ylim3d(ymin,ymax_plot_range)
ax.set_zlim3d(zmin,zmax_plot_range)

ax.grid(False)

ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')

ax.w_xaxis.set_pane_color((0.0, 1.0, 1.0, 0.15))
ax.w_yaxis.set_pane_color((0.0, 1.0, 0.0, 0.15))
ax.w_zaxis.set_pane_color((1.0, 1.0, 0.0, 1.0))

ax.view_init(elev=view_elev, azim=view_azim)

# axes labels:
axeslabelfontsize = 24
ax.set_xlabel('x',fontname=fontname,fontsize=axeslabelfontsize)
ax.set_ylabel('y',fontname=fontname,fontsize=axeslabelfontsize)
ax.set_zlabel('z',fontname=fontname,fontsize=axeslabelfontsize)

# plt.title((infileprefix.split('/'))[-1],fontname=fontname,fontsize=24) # make plot title the imported file name, without the file type and the parent folder names

# display surfaces
#Xsurf = np.arange( xstretch * xmin, xstretch * (xmax + 1), xstretch * (xmax-xmin)/10)
#Ysurf = np.arange( ystretch * ymin, ystretch * (ymax + 1), ystretch * (ymax-ymin)/10)
#Xsurf, Ysurf = np.meshgrid(Xsurf, Ysurf)
#Zsurf_top = 0*Xsurf + zstretch * zmax
#Zsurf_bottom = 0*Xsurf
#ax.plot_surface(Xsurf,Ysurf,Zsurf_bottom,color=(1.0,1.0,0,1.0))
#top_surf_opacity = 0.25
#ax.plot_surface(Xsurf,Ysurf,Zsurf_top,color=(1.0,1.0,0,top_surf_opacity))

if b_save_fig:
	plt.savefig(savefig_name, dpi=200)

if b_show_fig:
	plt.show()
