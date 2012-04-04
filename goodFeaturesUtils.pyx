# cython: profile=True
import math, numpy as np
cimport numpy as np

#*********************************************************************
#* _minEigenvalue
#*
#* Given the three distinct elements of the symmetric 2x2 matrix
#*                     [gxx gxy]
#*                     [gxy gyy],
#* Returns the minimum eigenvalue of the matrix.  
#*

cdef float _minEigenvalue(float gxx, float gxy, float gyy):
	cdef float sqrtTerm = ((gxx - gyy)*(gxx - gyy) + 4.*gxy*gxy) ** 0.5
	return (gxx + gyy - sqrtTerm)/2.

#*********************************************************************

cdef float SumGradientInWindow(int x,int y,int window_hh,int window_hw, np.ndarray[np.float32_t,ndim=2] cumSum):

	#Sum the gradients in the surrounding window with numpy
	cdef float a = cumSum[y-window_hh-1, x-window_hw-1]
	cdef float b = cumSum[y-window_hh-1, x+window_hw]
	cdef float c = cumSum[y+window_hh, x+window_hw]
	cdef float d = cumSum[y+window_hh, x-window_hw-1]

	return (c + a - b - d)

#**********************************************************************

def ScanImageForGoodFeatures(gradxArr, gradyArr, int borderx, int bordery, int window_hw, int window_hh, int nSkippedPixels):
	cdef float val, gxx, gxy, gyy
	cdef int x, y
	cdef int nrows = gradxArr.shape[0]
	cdef int ncols = gradxArr.shape[1]
	cdef int npoints = 0

	# For most of the pixels in the image, do ... 
	pointlistx,pointlisty,pointlistval = [], [], []
	

	cdef np.ndarray[np.float32_t,ndim=2] gradxxCumSum2 = np.power(gradxArr,2.).cumsum(axis=1).cumsum(axis=0)
	cdef np.ndarray[np.float32_t,ndim=2] gradyxCumSum2 = (gradxArr * gradyArr).cumsum(axis=1).cumsum(axis=0)
	cdef np.ndarray[np.float32_t,ndim=2] gradyyCumSum2 = np.power(gradyArr,2.).cumsum(axis=1).cumsum(axis=0)

	for y in range(bordery, nrows - bordery):
		for x in range(borderx, ncols - borderx):

			gxx = SumGradientInWindow(x,y,window_hh,window_hw,gradxxCumSum2)
			gxy = SumGradientInWindow(x,y,window_hh,window_hw,gradyxCumSum2)
			gyy = SumGradientInWindow(x,y,window_hh,window_hw,gradyyCumSum2)

			# Store the trackability of the pixel as the minimum
			# of the two eigenvalues 
			pointlistx.append(x)
			pointlisty.append(y)
			val = _minEigenvalue(gxx, gxy, gyy);
			
			pointlistval.append(int(val))
			npoints += 1
			x += nSkippedPixels + 1

		y += nSkippedPixels + 1

	return pointlistx,pointlisty,pointlistval

