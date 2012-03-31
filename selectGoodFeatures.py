
import math
from PIL import Image
from klt import *
from error import *
from convolve import *
from klt_util import *

class selectionMode:
	SELECTING_ALL = 1
	REPLACING_SOME = 2
KLT_verbose = 1

class kltState:
	KLT_TRACKED = 0
	KLT_NOT_FOUND = -1
	KLT_SMALL_DET = -2
	KLT_MAX_ITERATIONS = -3
	KLT_OOB = -4
	KLT_LARGE_RESIDUE = -5

#*********************************************************************

def _fillFeaturemap(x, y, featuremap, mindist, ncols, nrows):

	for iy in range(y - mindist,y + mindist + 1):
		for ix in range(x - mindist, x + mindist + 1):
			if ix >= 0 and ix < ncols and iy >= 0 and iy < nrows:
				featuremap[iy*ncols+ix] = True

	return featuremap


#*********************************************************************
#* _enforceMinimumDistance
#*
#* Removes features that are within close proximity to better features.
#*
#* INPUTS
#* featurelist:  A list of features.  The nFeatures property
#*               is used.
#*
#* OUTPUTS
#* featurelist:  Is overwritten.  Nearby "redundant" features are removed.
#*               Writes -1's into the remaining elements.
#*
#* RETURNS
#* The number of remaining features.
#*

def _enforceMinimumDistance(pointlist, featurelist, ncols, nrows, mindist, min_eigenvalue, overwriteAllFeatures):

	#int indx;          # Index into features 
	#int x, y, val;     # Location and trackability of pixel under consideration 
	#uchar *featuremap; # Boolean array recording proximity of features 
	#int *ptr;

	# Cannot add features with an eigenvalue less than one
	if min_eigenvalue < 1:  min_eigenvalue = 1

	# Allocate memory for feature map and clear it
	#featuremap = (uchar *) malloc(ncols * nrows * sizeof(uchar));
	#memset(featuremap, 0, ncols*nrows);
	featuremap = [False for i in range(ncols * nrows)]
	
	# Necessary because code below works with (mindist-1)
	mindist = mindist - 1

	# If we are keeping all old good features, then add them to the featuremap 
	if not overwriteAllFeatures:
		for indx, feat in enumerate(featurelist):
			if featurelist[indx].val >= 0:
				x = int(featurelist[indx].x)
				y = int(featurelist[indx].y)
				featuremap = _fillFeaturemap(x, y, featuremap, mindist, ncols, nrows)		

	# For each feature point, in descending order of importance, do ... 
	indx = 0
	pointlistIndx = 0
	while True:

		# If we can't add all the points, then fill in the rest
		#  of the featurelist with -1's */
		if pointlistIndx >= len(pointlist):
			while indx < len(featurelist):
				if overwriteAllFeatures and featurelist[indx].val < 0:
					featurelist[indx].x   = -1
					featurelist[indx].y   = -1
					featurelist[indx].val = kltState.KLT_NOT_FOUND
					featurelist[indx].aff_img = None
					featurelist[indx].aff_img_gradx = None
					featurelist[indx].aff_img_grady = None
					featurelist[indx].aff_x = -1.0
					featurelist[indx].aff_y = -1.0
					featurelist[indx].aff_Axx = 1.0
					featurelist[indx].aff_Ayx = 0.0
					featurelist[indx].aff_Axy = 0.0
					featurelist[indx].aff_Ayy = 1.0
			  	indx = indx + 1
			break

		x = pointlist[pointlistIndx]
		y = pointlist[pointlistIndx+1]
		val = pointlist[pointlistIndx+2]
		#print pointlistIndx, val, len(pointlist)

		pointlistIndx += 3
		
		# Ensure that feature is in-bounds 
		assert x >= 0
		assert x < ncols
		assert y >= 0
		assert y < nrows
	
		while not overwriteAllFeatures and indx < len(featurelist) and featurelist[indx].val >= 0:
			indx = indx + 1

		if indx >= len(featurelist): break

		# If no neighbor has been selected, and if the minimum
		#   eigenvalue is large enough, then add feature to the current list 
		if not featuremap[y*ncols+x] and val >= min_eigenvalue:
			featurelist[indx].x   = x
			featurelist[indx].y   = y
			featurelist[indx].val = int(val)
			featurelist[indx].aff_img = None
			featurelist[indx].aff_img_gradx = None
			featurelist[indx].aff_img_grady = None
			featurelist[indx].aff_x = -1.0
			featurelist[indx].aff_y = -1.0
			featurelist[indx].aff_Axx = 1.0
			featurelist[indx].aff_Ayx = 0.0
			featurelist[indx].aff_Axy = 0.0
			featurelist[indx].aff_Ayy = 1.0
			indx = indx + 1

			# Fill in surrounding region of feature map, but
			#    make sure that pixels are in-bounds */
			featuremap = _fillFeaturemap(x, y, featuremap, mindist, ncols, nrows);

	return featurelist


#*********************************************************************
#* _sortPointList
#*

def _sortPointList(pointlist):

	#This is probably grossly inefficient. Better to use numpy?
	ptx = [pointlist[i] for i in range(0,len(pointlist),3)]
	pty = [pointlist[i] for i in range(1,len(pointlist),3)]
	pteig = [pointlist[i] for i in range(2,len(pointlist),3)]

	li = zip(pteig, ptx, pty)
	li.sort()
	li.reverse()
	out = []
	for item in li:
		out.extend((item[1],item[2],item[0]))
	return out

#*********************************************************************
#* _minEigenvalue
#*
#* Given the three distinct elements of the symmetric 2x2 matrix
#*                     [gxx gxy]
#*                     [gxy gyy],
#* Returns the minimum eigenvalue of the matrix.  
#*

def _minEigenvalue(gxx, gxy, gyy):
	return float((gxx + gyy - math.sqrt((gxx - gyy)*(gxx - gyy) + 4*gxy*gxy))/2.0)


#*********************************************************************

def _KLTSelectGoodFeatures(tc,img,nFeatures,mode):

	featurelist = [KLT_Feature() for i in range(nFeatures)]
	#_KLT_FloatImage floatimg, gradx, grady;
	#int window_hw, window_hh
	#int *pointlist
	overwriteAllFeatures = (mode == selectionMode.SELECTING_ALL)
	floatimages_created = False
	ncols, nrows = img.size

	# Check window size (and correct if necessary) 
	if tc.window_width % 2 != 1:
		tc.window_width = tc.window_width+1
		KLTWarning("Tracking context's window width must be odd.  Changing to {0}.\n".format(tc.window_width))

	if tc.window_height % 2 != 1:
		tc.window_height = tc.window_height+1
		KLTWarning("Tracking context's window height must be odd.  Changing to {0}.\n".format(tc.window_height))
	
	if tc.window_width < 3:
		tc.window_width = 3
		KLTWarning("Tracking context's window width must be at least three.  \nChanging to %d.\n".format(tc.window_width))

	if tc.window_height < 3:
		tc.window_height = 3
		KLTWarning("Tracking context's window height must be at least three.  \nChanging to %d.\n".format(tc.window_height))
	
	window_hw = tc.window_width/2 
	window_hh = tc.window_height/2

	# Create pointlist, which is a simplified version of a featurelist, 
	# for speed.  Contains only integer locations and values. 
	#pointlist = [0 for i in range(ncols * nrows * 3)]

	# Create temporary images, etc. 
	if mode == selectionMode.REPLACING_SOME and tc.sequentialMode and tc.pyramid_last != None:
		floatimg = tc.pyramid_last.img[0]
		gradx = tc.pyramid_last_gradx.img[0]
		grady = tc.pyramid_last_grady.img[0]
		assert gradx != None
		assert grady != None
	else:
		floatimages_created = True
		floatimg = Image.new("F", img.size)
		gradx    = Image.new("F", img.size)
		grady    = Image.new("F", img.size)
		if tc.smoothBeforeSelecting:
			#_KLT_FloatImage tmpimg;
			#tmpimg = Image.new("F", img.size)
			tmpimg = img.convert("F")
			floatimg = KLTComputeSmoothedImage(tmpimg, KLTComputeSmoothSigma(tc))
			#_KLTFreeFloatImage(tmpimg)
		else:
			floatimg = img.convert("F")

		# Compute gradient of image in x and y direction 
		gradx, grady = KLTComputeGradients(floatimg, tc.grad_sigma)
	
	
	# Write internal images 
	if tc.writeInternalImages:
		floatimg.save("kltimg_sgfrlf.pgm")
		gradx.save("kltimg_sgfrlf_gx.pgm")
		grady.save("kltimg_sgfrlf_gy.pgm")

	# Compute trackability of each image pixel as the minimum
	#   of the two eigenvalues of the Z matrix 
	
	#register float gx, gy;
	#register float gxx, gxy, gyy;
	#register int xx, yy;
	#register int *ptr;
	#float val;
	#unsigned int limit = 1;
	borderx = tc.borderx;	# Must not touch cols 
	bordery = tc.bordery;	# lost by convolution 
	#int x, y;
	#int i;
	
	if borderx < window_hw: borderx = window_hw
	if bordery < window_hh: bordery = window_hh

	# Find largest value of an int 
	#for (i = 0 ; i < sizeof(int) ; i++)  limit *= 256;
	#limit = limit/2 - 1;
		
	gradxl = gradx.load()
	gradyl = grady.load()

	# For most of the pixels in the image, do ... 
	pointlist = []
	npoints = 0
	for y in range(bordery, nrows - bordery):
		for x in range(borderx, ncols - borderx):
			# Sum the gradients in the surrounding window 
			gxx = 0
			gxy = 0
			gyy = 0
			for yy in range(y-window_hh, y+window_hh+1):
				for xx in range(x-window_hw, x+window_hw+1):
					#gx = *(gradx->data + ncols*yy+xx);
					gx = gradxl[xx,yy]
					#gy = *(grady->data + ncols*yy+xx);
					gy = gradyl[xx,yy]
					gxx += gx * gx;
					gxy += gx * gy;
					gyy += gy * gy;

			# Store the trackability of the pixel as the minimum
			# of the two eigenvalues 
			pointlist.append(x)
			pointlist.append(y)
			val = _minEigenvalue(gxx, gxy, gyy);
			#if val > limit:
				#TWarning("(_KLTSelectGoodFeatures) minimum eigenvalue %f is "
				#           "greater than the capacity of an int; setting "
				#           "to maximum value", val);
				#val = (float) limit;
			
			pointlist.append(int(val))
			npoints = npoints + 1
			x += tc.nSkippedPixels + 1

		y += tc.nSkippedPixels + 1
			
	# Sort the features 
	pointlist = _sortPointList(pointlist)

	#print pointlist

	# Check tc.mindist 
	if tc.mindist < 0:
		KLTWarning("(_KLTSelectGoodFeatures) Tracking context field tc.mindist is negative ({0}); setting to zero".format(tc.mindist))
		tc.mindist = 0;

	# Enforce minimum distance between features 
	_enforceMinimumDistance(pointlist, \
	  featurelist, \
	  ncols, nrows, \
	  tc.mindist, \
	  tc.min_eigenvalue, \
	  overwriteAllFeatures)

	# Free memory 
	#  free(pointlist);
	#  if (floatimages_created)  {
	#    _KLTFreeFloatImage(floatimg);
	#    _KLTFreeFloatImage(gradx);
	#    _KLTFreeFloatImage(grady);
	#  }

	return featurelist

#*********************************************************************
#* KLTSelectGoodFeatures
#*
#* Main routine, visible to the outside.  Finds the good features in
#* an image.  
#* 
#* INPUTS
#* tc:	Contains parameters used in computation (size of image,
#*        size of window, min distance b/w features, sigma to compute
#*        image gradients, # of features desired).
#* img:	Pointer to the data of an image (probably unsigned chars).
#* 
#* OUTPUTS
#* features:	List of features.  The member nFeatures is computed.
#*

def KLTSelectGoodFeatures(tc, img, nFeatures):
	
	ncols, nrows = img.size

	#int ncols, int nrows,
	if KLT_verbose >= 1:
		print "(KLT) Selecting the {0} best features from a {1} by {2} image...  ".format(nFeatures, ncols, nrows)

	fl = _KLTSelectGoodFeatures(tc, img, nFeatures, selectionMode.SELECTING_ALL)

	if KLT_verbose >= 1:
		print "\n\t{0} features found.\n".format(KLTCountRemainingFeatures(fl))
    	if tc.writeInternalImages:
		print "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n"

	return fl

