# cython: profile=False
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

	assert ix - hw >= 0 and iy - hh >= 0 and ix + hw + 2 <= ncols and iy + hh + 2 <= nrows

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

	i = 0 #All done, but this line makes cython profiling easier to read

#*********************************************************************
#* _computeIntensityDifference
#*
#* Given two images and the window center in both images,
#* aligns the images wrt the window and computes the difference 
#* between the two overlaid images.
#*

cdef _computeIntensityDifference(np.ndarray[np.float32_t,ndim=2] img1Patch,   # images 
	np.ndarray[np.float32_t,ndim=2] img2,
	float x2, 
	float y2,     # center of window in 2nd img
	np.ndarray[np.float32_t,ndim=2] workingPatch, # temporary memory for patch storage, size determines window size
	np.ndarray[np.float32_t,ndim=1] out):

	cdef int hw = workingPatch.shape[1]/2
	cdef int hh = workingPatch.shape[0]/2
	cdef float g1, g2
	cdef int i, j, ind = 0

	#imgdiff = []
	#imgl1 = img1.load()
	#imgl2 = img2.load()

	extractImagePatchOptimised(img2, x2, y2, workingPatch)

	# Compute values
	for j in range(-hh, hh + 1):
		for i in range(-hw, hw + 1):
			g1 = img1Patch[j + hh, i + hw]
			g2 = workingPatch[j + hh, i + hw]
			#imgdiff.append(g1 - g2)
			out[ind] = g1 - g2
			ind += 1

	return None

def computeIntensityDifference(np.ndarray[np.float32_t,ndim=2] img1Patch,   # images 
	np.ndarray[np.float32_t,ndim=2] img2,
	float x2, 
	float y2,     # center of window in 2nd img
	np.ndarray[np.float32_t,ndim=2] workingPatch,
	np.ndarray[np.float32_t,ndim=1] out): # temporary memory for patch storage, size determines window size

	return _computeIntensityDifference(img1Patch, img2, x2, y2, workingPatch, out)

#*********************************************************************
#* _computeGradientSum
#*
#* Given two gradients and the window center in both images,
#* aligns the gradients wrt the window and computes the sum of the two 
#* overlaid gradients.
#*

cdef _computeGradientSum(np.ndarray[np.float32_t,ndim=2] img1GradxPatch,  # gradient images
	np.ndarray[np.float32_t,ndim=2] gradx2,
	float x2, float y2,      # center of window in 2nd img
	np.ndarray[np.float32_t,ndim=2] workingPatch, # temporary memory for patch storage, size determines window size
	np.ndarray[np.float32_t,ndim=2] out,
	int row): 

	cdef int hw = workingPatch.shape[1]/2
	cdef int hh = workingPatch.shape[0]/2
	cdef float g1, g2
	cdef int i, j
	#gradx, grady = [], []

	extractImagePatchOptimised(gradx2, x2, y2, workingPatch)

	# Compute values
	for j in range(workingPatch.shape[0]):
		for i in range(workingPatch.shape[1]):
			g1 = img1GradxPatch[j, i]
			g2 = workingPatch[j, i]

			out[j*workingPatch.shape[0] + i, row] = - g1 - g2

def computeGradientSum(np.ndarray[np.float32_t,ndim=2] img1GradxPatch,  # gradient images
	np.ndarray[np.float32_t,ndim=2] gradx2,
	float x2, float y2,      # center of window in 2nd img
	np.ndarray[np.float32_t,ndim=2] workingPatch, # temporary memory for patch storage, size determines window size
	np.ndarray[np.float32_t,ndim=2] out,
	int row):

	return _computeGradientSum(img1GradxPatch,
		gradx2,
		x2, y2,
		workingPatch,
		out,
		row)

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

cdef _compute2by1ErrorVector(np.ndarray[np.float32_t,ndim=1] imgdiff,
	np.ndarray[np.float32_t,ndim=2] jacobian,
	int width, # size of window
	int height,
	float step_factor,
	np.ndarray[np.float32_t,ndim=1] out): # 2.0 comes from equations, 1.0 seems to avoid overshooting

	#cdef np.ndarray[np.float32_t,ndim=1] gradx = - jacobian[:,0]
	#cdef np.ndarray[np.float32_t,ndim=1] grady = - jacobian[:,1]

	# Compute values
	cdef float ex = 0.
	cdef float ey = 0.
	cdef int ind = 0
	cdef int i = 0
	cdef float diff = 0.

	for i in range(width * height):
		diff = imgdiff[ind]
		ex += - diff * jacobian[ind,0]
		ey += - diff * jacobian[ind,1]
		ind += 1

	ex *= step_factor
	ey *= step_factor

	out[0] = ex
	out[1] = ey

#*********************************************************************
#* _compute2by2GradientMatrix
#*
#*

cdef int _compute2by2GradientMatrix(np.ndarray[np.float32_t,ndim=2] jacobian,
	int width,   # size of window
	int height,
	np.ndarray[np.float32_t,ndim=2] out):

	#cdef np.ndarray[np.float32_t,ndim=1] gradx = - jacobian[:,0]
	#cdef np.ndarray[np.float32_t,ndim=1] grady = - jacobian[:,1]

	# Compute values 
	cdef float gx, gy
	out[0,0] = 0.
	out[1,0] = 0.
	out[0,1] = 0.
	out[1,1] = 0.
	cdef int ind = 0, i

	for i in range(width * height):
		gx = - jacobian[ind,0]
		gy = - jacobian[ind,1]
		out[0,0] += gx*gx;
		out[1,0] += gx*gy;
		out[1,1] += gy*gy;
		ind += 1

	out[0,1] = out[1,0]
	return 1 

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

cdef _solveEquation(np.ndarray[np.float32_t,ndim=2] gradientMatrix,
	np.ndarray[np.float32_t,ndim=1] errorMatrix,
	float small,
	np.ndarray[np.float32_t,ndim=1] predictedMotion):

	cdef float gxx = gradientMatrix[0,0]
	cdef float gxy = gradientMatrix[0,1]
	cdef float gyy = gradientMatrix[1,1]
	cdef float ex = errorMatrix[0]
	cdef float ey = errorMatrix[1]

	cdef float det = gxx*gyy - gxy*gxy, dx, dy
	
	if det < small: 
		predictedMotion[0] = 0.
		predictedMotion[1] = 0.
		return kltState.KLT_SMALL_DET

	dx = (gyy*ex - gxy*ey)/det
	dy = (gxx*ey - gxy*ex)/det
	predictedMotion[0] = dx
	predictedMotion[1] = dy
	return kltState.KLT_TRACKED

def minFunc(np.ndarray[double,ndim=1] xData, 
	np.ndarray[np.float32_t,ndim=2] img1Patch, 
	np.ndarray[np.float32_t,ndim=2] img1GradxPatch, 
	np.ndarray[np.float32_t,ndim=2] img1GradyPatch, 
	np.ndarray[np.float32_t,ndim=2] img2, 
	np.ndarray[np.float32_t,ndim=2] workingPatch, 
	np.ndarray[np.float32_t,ndim=2] jacobianMem, 
	int lightInsensitive,
	np.ndarray[np.float32_t,ndim=2] gradx2, 
	np.ndarray[np.float32_t,ndim=2] grady2):

	cdef float x2 = xData[0]
	cdef float y2 = xData[1]

	#print img1, img2, x1, y1, width, height
	if lightInsensitive:
		raise Exception("Not implemented")
		#imgdiff = _computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, x2, y2, workingPatch)
	else:
		_computeIntensityDifference(img1Patch, img2, x2, y2, workingPatch, jacobianMem[:,0])

	#print "test", x2, y2, np.array(imgdiff).sum()
	return jacobianMem[:,0]

def jacobian(np.ndarray[double,ndim=1] xData, 
	np.ndarray[np.float32_t,ndim=2] img1Patch, 
	np.ndarray[np.float32_t,ndim=2] img1GradxPatch, 
	np.ndarray[np.float32_t,ndim=2] img1GradyPatch, 
	np.ndarray[np.float32_t,ndim=2] img2, 
	np.ndarray[np.float32_t,ndim=2] workingPatch, 
	np.ndarray[np.float32_t,ndim=2] jacobianMem, 
	int lightInsensitive,
	np.ndarray[np.float32_t,ndim=2] gradx2, 
	np.ndarray[np.float32_t,ndim=2] grady2):

	cdef float x2 = xData[0]
	cdef float y2 = xData[1]

	#print img1, img2, x1, y1, width, height
	if lightInsensitive:
		raise Exception("Not implemented")
		#gradx, grady = _computeGradientSumLightingInsensitive(gradx1, grady1, gradx, grady2, img1, img2, x1, y1, x2, y2, workingPatch, jacobianMem)
	else:
		_computeGradientSum(img1GradxPatch, gradx2, x2, y2, workingPatch, jacobianMem, 0)
		_computeGradientSum(img1GradyPatch, grady2, x2, y2, workingPatch, jacobianMem, 1)

	return jacobianMem


#*********************************************************************

def trackFeatureIterateCKLT(float x2, 
	float y2, 
	np.ndarray[np.float32_t,ndim=2] img1GradxPatch, 
	np.ndarray[np.float32_t,ndim=2] img1GradyPatch, 
	np.ndarray[np.float32_t,ndim=2] img1Patch, 
	np.ndarray[np.float32_t,ndim=2] img2, 
	np.ndarray[np.float32_t,ndim=2] gradx2, 
	np.ndarray[np.float32_t,ndim=2] grady2, 
	tc):

	cdef int width = tc.window_width # size of window
	cdef int height = tc.window_height 
	cdef int lighting_insensitive = tc.lighting_insensitive # whether to normalize for gain and bias 
	cdef float step_factor = tc.step_factor # 2.0 comes from equations, 1.0 seems to avoid overshooting
	cdef float small = tc.min_determinant # determinant threshold for declaring KLT_SMALL_DET 
	cdef float th = tc.min_displacement # displacement threshold for stopping             
	cdef int max_iterations = tc.max_iterations 

	cdef int iteration = 0
	cdef float one_plus_eps = 1.001   # To prevent rounding errors 
	cdef int hw = width/2
	cdef int hh = height/2
	cdef int nc = img2.shape[1]
	cdef int nr = img2.shape[0]
	cdef np.ndarray[np.float32_t,ndim=2] workingPatch = np.empty((height, width), np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] imgdiff = np.empty((workingPatch.size), np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] jacobian = np.empty((workingPatch.size,2), np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] gradientMatrix = np.empty((2,2), np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] errorMatrix = np.empty((2,), np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] predictedMotion = np.empty((2,), np.float32)

	# Iteratively update the window position 
	while True:

		# If out of bounds, exit loop 
		if x2-hw < 0. or nc-(x2+hw) < one_plus_eps or \
			y2-hh < 0. or nr-(y2+hh) < one_plus_eps:
			status = kltState.KLT_OOB
			break

		# Compute gradient and difference windows 
		if lighting_insensitive:
			raise Exception("Not implemented")
			#imgdiff = _computeIntensityDifferenceLightingInsensitive(img1Patch, img2, x2, y2, workingPatch)
			#gradx, grady = computeGradientSumLightingInsensitive(gradx1, grady1, gradx, grady2, img1, img2, x1, y1, x2, y2, workingPatch)
		else:
			_computeIntensityDifference(img1Patch, img2, x2, y2, workingPatch, imgdiff)

			_computeGradientSum(img1GradxPatch, gradx2, x2, y2, workingPatch, jacobian, 0)
			_computeGradientSum(img1GradyPatch, grady2, x2, y2, workingPatch, jacobian, 1)

		# Use these windows to construct matrices 
		_compute2by2GradientMatrix(jacobian, width, height, gradientMatrix)
		_compute2by1ErrorVector(imgdiff, jacobian, width, height, step_factor, errorMatrix)

		# Using matrices, solve equation for new displacement */
		status = _solveEquation(gradientMatrix, errorMatrix, small, predictedMotion)
		if status == kltState.KLT_SMALL_DET: break

		x2 += predictedMotion[0]
		y2 += predictedMotion[1]
		iteration += 1

		if not ((abs(predictedMotion[0])>=th or abs(predictedMotion[1])>=th) and iteration < max_iterations): break

	return x2, y2, status, iteration

