# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
from klt import *
import scipy.optimize

#*********************************************************************
#* interpolate
#* 
#* Given a point (x,y) in an image, computes the bilinear interpolated 
#* gray-level value of the point in the image.  
#*

def interpolate(float x, float y, np.ndarray[np.float32_t,ndim=2] img):

	cdef int xt = int(x)  # coordinates of top-left corner 
	cdef int yt = int(y)
	cdef float ax = x - xt
	cdef float ay = y - yt
	cdef float out

	cdef int ncols = img.shape[1]
	cdef int nrows = img.shape[0]

	#_DNDEBUG = False
	#if not _DNDEBUG:
	#	if (xt<0 or yt<0 or xt>=ncols-1 or yt>=nrows-1):
	#		print "(xt,yt)=({0},{1})  imgsize=({2},{3})\n(x,y)=({4},{5})  (ax,ay)=({6},{7})".format(
	#			xt, yt, ncols, nrows, x, y, ax, ay)

	if xt < 0 or yt < 0 or xt > ncols - 2 or yt > nrows - 2:
		return 0.

	#assert xt >= 0 and yt >= 0 and xt <= ncols - 2 and yt <= nrows - 2

	out = (1-ax) * (1-ay) * img[yt,xt] + \
		ax   * (1-ay) * img[yt,xt+1] + \
		(1-ax) *   ay   * img[yt+1,xt] + \
		ax   *   ay   * img[yt+1,xt+1]
	return out


#*********************************************************************
#* _computeIntensityDifference
#*
#* Given two images and the window center in both images,
#* aligns the images wrt the window and computes the difference 
#* between the two overlaid images.
#*

def _computeIntensityDifference(img1Patch,   # images 
	np.ndarray[np.float32_t,ndim=2] img2,
	#float x1, 
	#float y1,     # center of window in 1st img
	float x2, 
	float y2,     # center of window in 2nd img
	int width, 
	int height):  # size of window

	cdef int hw = width/2
	cdef int hh = height/2
	cdef float g1, g2
	cdef int i, j

	img2Patch = np.empty((height, width))
	for j in range(-hh, hh + 1):
		for i in range(-hw, hw + 1):
			img2Patch[j+hh,i+hw] = interpolate(x2+i, y2+j, img2)

	#print img1Patch.shape, img2Patch.shape
	assert img1Patch.shape == img2Patch.shape

	diffImg = img1Patch - img2Patch
	diffImg = diffImg.reshape((diffImg.shape[0] * diffImg.shape[1]))

	return diffImg



#*********************************************************************
#* _computeGradientSum
#*
#* Given two gradients and the window center in both images,
#* aligns the gradients wrt the window and computes the sum of the two 
#* overlaid gradients.
#*

def _computeGradientSum(np.ndarray[np.float32_t,ndim=2] gradx1,  # gradient images
	np.ndarray[np.float32_t,ndim=2] gradx2,
	float x1, float y1,      # center of window in 1st img
	float x2, float y2,      # center of window in 2nd img
	int width, int height):   # size of window

	cdef int hw = width/2
	cdef int hh = height/2
	cdef float g1, g2
	cdef int i, j
	gradx = []

	img1GradxPatch = np.empty((height, width))
	for j in range(-hh, hh + 1):
		for i in range(-hw, hw + 1):
			img1GradxPatch[j+hh,i+hw] = interpolate(x1+i, y1+j, gradx1)

	img2Patch = np.empty((height, width))
	for j in range(-hh, hh + 1):
		for i in range(-hw, hw + 1):
			img2Patch[j+hh,i+hw] = interpolate(x2+i, y2+j, gradx2)

	sumImg = img1GradxPatch + img2Patch
	sumImg = sumImg.reshape((sumImg.shape[0] * sumImg.shape[1]))

	return sumImg

#*********************************************************************
#* _computeIntensityDifferenceLightingInsensitive
#*
#* Given two images and the window center in both images,
#* aligns the images wrt the window and computes the difference 
#* between the two overlaid images; normalizes for overall gain and bias.
#*

#static void _computeIntensityDifferenceLightingInsensitive(
#  _KLT_FloatImage img1,   /* images */
#  _KLT_FloatImage img2,
#  float x1, float y1,     /* center of window in 1st img */
#  float x2, float y2,     /* center of window in 2nd img */
#  int width, int height,  /* size of window */
#  _FloatWindow imgdiff)   /* output */
#{
#  register int hw = width/2, hh = height/2;
#  float g1, g2, sum1_squared = 0, sum2_squared = 0;
#  register int i, j;
#  
#  float sum1 = 0, sum2 = 0;
#  float mean1, mean2,alpha,belta;
#  /* Compute values */
#  for (j = -hh ; j <= hh ; j++)
#    for (i = -hw ; i <= hw ; i++)  {
#      g1 = trackFeaturesUtils.interpolate(x1+i, y1+j, img1);
#      g2 = trackFeaturesUtils.interpolate(x2+i, y2+j, img2);
#      sum1 += g1;    sum2 += g2;
#      sum1_squared += g1*g1;
#      sum2_squared += g2*g2;
#   }
#  mean1=sum1_squared/(width*height);
#  mean2=sum2_squared/(width*height);
#  alpha = (float) sqrt(mean1/mean2);
#  mean1=sum1/(width*height);
#  mean2=sum2/(width*height);
#  belta = mean1-alpha*mean2;
#
#  for (j = -hh ; j <= hh ; j++)
#    for (i = -hw ; i <= hw ; i++)  {
#      g1 = trackFeaturesUtils.interpolate(x1+i, y1+j, img1);
#      g2 = trackFeaturesUtils.interpolate(x2+i, y2+j, img2);
#      *imgdiff++ = g1- g2*alpha-belta;
#    } 
#}


#*********************************************************************
#* _computeGradientSumLightingInsensitive
#*
#* Given two gradients and the window center in both images,
#* aligns the gradients wrt the window and computes the sum of the two 
#* overlaid gradients; normalizes for overall gain and bias.
#*

#static void _computeGradientSumLightingInsensitive(
#  _KLT_FloatImage gradx1,  /* gradient images */
#  _KLT_FloatImage grady1,
#  _KLT_FloatImage gradx2,
#  _KLT_FloatImage grady2,
#  _KLT_FloatImage img1,   /* images */
#  _KLT_FloatImage img2,
# 
#  float x1, float y1,      /* center of window in 1st img */
#  float x2, float y2,      /* center of window in 2nd img */
#  int width, int height,   /* size of window */
#  _FloatWindow gradx,      /* output */
#  _FloatWindow grady)      /*   " */
#{
#  register int hw = width/2, hh = height/2;
#  float g1, g2, sum1_squared = 0, sum2_squared = 0;
#  register int i, j;
#  
#  float sum1 = 0, sum2 = 0;
#  float mean1, mean2, alpha;
#  for (j = -hh ; j <= hh ; j++)
#    for (i = -hw ; i <= hw ; i++)  {
#      g1 = trackFeaturesUtils.interpolate(x1+i, y1+j, img1);
#      g2 = trackFeaturesUtils.interpolate(x2+i, y2+j, img2);
#      sum1_squared += g1;    sum2_squared += g2;
#    }
#  mean1 = sum1_squared/(width*height);
#  mean2 = sum2_squared/(width*height);
#  alpha = (float) sqrt(mean1/mean2);
#  
#  /* Compute values */
#  for (j = -hh ; j <= hh ; j++)
#    for (i = -hw ; i <= hw ; i++)  {
#      g1 = trackFeaturesUtils.interpolate(x1+i, y1+j, gradx1);
#      g2 = trackFeaturesUtils.interpolate(x2+i, y2+j, gradx2);
#      *gradx++ = g1 + g2*alpha;
#      g1 = trackFeaturesUtils.interpolate(x1+i, y1+j, grady1);
#      g2 = trackFeaturesUtils.interpolate(x2+i, y2+j, grady2);
#      *grady++ = g1+ g2*alpha;
#    }  
#}

#*********************************************************************
#* _compute2by1ErrorVector
#*
#*

def _compute2by1ErrorVector(imgdiff,
	gradx,
	grady,
	width, # size of window
	height,
	step_factor): # 2.0 comes from equations, 1.0 seems to avoid overshooting

	# Compute values
	ex = 0.
	ey = 0.
	ind = 0
	for i in range(width * height):
		diff = imgdiff[ind]
		ex += diff * gradx[ind]
		ey += diff * grady[ind]
		ind += 1

	ex *= step_factor
	ey *= step_factor

	return ex, ey

#*********************************************************************
#* _compute2by2GradientMatrix
#*
#*

def _compute2by2GradientMatrix(gradx, grady,
	width,   # size of window */
	height):

	# Compute values 
	gxx = 0.0
	gxy = 0.0
	gyy = 0.0
	ind = 0
	for i in range(width * height):
		gx = gradx[ind]
		gy = grady[ind]
		gxx += gx*gx;
		gxy += gx*gy;
		gyy += gy*gy;
		ind += 1

	return gxx, gxy, gyy
	



#*********************************************************************
#* _solveEquation
#*
#* Solves the 2x2 matrix equation
#*         [gxx gxy] [dx] = [ex]
#*         [gxy gyy] [dy] = [ey]
#* for dx and dy.
#*
#* Returns KLT_TRACKED on success and KLT_SMALL_DET on failure
#*

def _solveEquation(gxx, gxy, gyy,
	ex, ey,
	small):

	det = gxx*gyy - gxy*gxy
	
	if det < small: return kltState.KLT_SMALL_DET, None, None

	dx = (gyy*ex - gxy*ey)/det
	dy = (gxx*ey - gxy*ex)/det
	return kltState.KLT_TRACKED, dx, dy

def minFunc(xData, img1Patch, img1, img2, x1, y1, width, height, tc, gradx1, grady1, gradx2, grady2):
	x2, y2 = xData

	#print img1, img2, x1, y1, width, height
	if tc.lighting_insensitive:
		raise Exception("Not implemented")
		#imgdiff = _computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, x2, y2, width, height)
	else:
		imgdiff = _computeIntensityDifference(img1Patch, img2, x2, y2, width, height)

	#print "test", x2, y2, np.array(imgdiff).sum()
	return imgdiff

def jacobian(xData, img1Patch, img1, img2, x1, y1, width, height, tc, gradx1, grady1, gradx2, grady2):
	x2, y2 = xData
	#print img1, img2, x1, y1, width, height
	if tc.lighting_insensitive:
		raise Exception("Not implemented")
		#gradx, grady = _computeGradientSumLightingInsensitive(gradx1, grady1, gradx, grady2, img1, img2, x1, y1, x2, y2, width, height)
	else:
		gradx = _computeGradientSum(gradx1, gradx2, x1, y1, x2, y2, width, height)
		grady = _computeGradientSum(grady1, grady2, x1, y1, x2, y2, width, height)
	out = - np.array([gradx, grady]).transpose()
	#print out.shape
	
	return out

