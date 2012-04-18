# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
from klt import *
import scipy.optimize
import scipy.ndimage

#*********************************************************************

def extractImagePatchSlow(np.ndarray[np.float32_t,ndim=2] img, float x, float y, int height, int width):

	patch = np.empty((height, width), np.float32)
	extractImagePatchOptimised(img, x, y, patch)
	return patch

cdef extractImagePatchOptimised(np.ndarray[np.float32_t,ndim=2] img, float x, float y, np.ndarray[np.float32_t,ndim=2] out):

	cdef int i, j, vx, vy
	cdef int ix = int(x)
	cdef int iy = int(y)
	cdef int patchCols = out.shape[1]
	cdef int patchRows = out.shape[0]
	cdef int hh = out.shape[0] / 2
	cdef int hw = out.shape[1] / 2
	cdef float val
	cdef float ax = x - int(x) #Get decimal part of x and y
	cdef float ay = y - int(y)
	cdef int ncols = img.shape[1]
	cdef int nrows = img.shape[0]

	assert ix - hw >= 0 and iy - hh >= 0 and ix + hw + 1 <= ncols - 2 and iy + hh + 1 <= nrows - 2

	# Compute values
	for j in range(patchCols):
		for i in range(patchRows):

			vx = ix+i-hw
			vy = iy+j-hh

			val = (1.-ax) * (1.-ay) * img[vy,vx] + \
				ax   * (1.-ay) * img[vy,vx+1] + \
				(1.-ax) *   ay   * img[vy+1,vx] + \
				ax   *   ay   * img[vy+1,vx+1]

			out[j,i] = val

	i = 0

#*********************************************************************
#* _computeIntensityDifference
#*
#* Given two images and the window center in both images,
#* aligns the images wrt the window and computes the difference 
#* between the two overlaid images.
#*

def _computeIntensityDifference(img1Patch,   # images 
	np.ndarray[np.float32_t,ndim=2] img2,
	float x2, 
	float y2,     # center of window in 2nd img
	workingPatch):  # temporary memory for patch storage, size determines window size

	cdef int hw = workingPatch.shape[1]/2
	cdef int hh = workingPatch.shape[0]/2
	cdef float g1, g2
	cdef int i, j

	imgdiff = []
	#imgl1 = img1.load()
	#imgl2 = img2.load()

	extractImagePatchOptimised(img2, x2, y2, workingPatch)

	# Compute values
	for j in range(-hh, hh + 1):
		for i in range(-hw, hw + 1):
			g1 = img1Patch[j + hh, i + hw]
			g2 = workingPatch[j + hh, i + hw]
			imgdiff.append(g1 - g2)
	
	return imgdiff

#*********************************************************************
#* _computeGradientSum
#*
#* Given two gradients and the window center in both images,
#* aligns the gradients wrt the window and computes the sum of the two 
#* overlaid gradients.
#*

def _computeGradientSum(img1GradxPatch,  # gradient images
	np.ndarray[np.float32_t,ndim=2] gradx2,
	float x2, float y2,      # center of window in 2nd img
	workingPatch):   # temporary memory for patch storage, size determines window size

	cdef int hw = workingPatch.shape[1]/2
	cdef int hh = workingPatch.shape[0]/2
	cdef float g1, g2
	cdef int i, j
	gradx, grady = [], []

	extractImagePatchOptimised(gradx2, x2, y2, workingPatch)

	# Compute values
	for j in range(-hh, hh + 1):
		for i in range(-hw, hw + 1):
			g1 = img1GradxPatch[j+hh, i+hw]
			g2 = workingPatch[j+hh, i+hw]
			gradx.append(g1 + g2)

	return gradx

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

def minFunc(xData, img1Patch, img1GradxPatch, img1GradyPatch, img2, workingPatch, tc, gradx2, grady2):
	x2, y2 = xData

	#print img1, img2, x1, y1, width, height
	if tc.lighting_insensitive:
		raise Exception("Not implemented")
		#imgdiff = _computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, x2, y2, workingPatch)
	else:
		imgdiff = _computeIntensityDifference(img1Patch, img2, x2, y2, workingPatch)

	#print "test", x2, y2, np.array(imgdiff).sum()
	return imgdiff

def jacobian(xData, img1Patch, img1GradxPatch, img1GradyPatch, img2, workingPatch, tc, gradx2, grady2):
	x2, y2 = xData

	#print img1, img2, x1, y1, width, height
	if tc.lighting_insensitive:
		raise Exception("Not implemented")
		#gradx, grady = _computeGradientSumLightingInsensitive(gradx1, grady1, gradx, grady2, img1, img2, x1, y1, x2, y2, workingPatch)
	else:
		gradx = _computeGradientSum(img1GradxPatch, gradx2, x2, y2, workingPatch)
		grady = _computeGradientSum(img1GradyPatch, grady2, x2, y2, workingPatch)
	out = - np.array([gradx, grady]).transpose()
	#print out.shape
	
	return out


