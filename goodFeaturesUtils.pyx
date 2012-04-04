# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
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

def ScanImageForGoodFeatures(np.ndarray[np.float32_t,ndim=2] gradxArr, 
	np.ndarray[np.float32_t,ndim=2] gradyArr, 
	int borderx, int bordery, int window_hw, int window_hh, int nSkippedPixels):

	cdef float val, gxx, gxy, gyy
	cdef int x, y
	cdef int nrows = gradxArr.shape[0]
	cdef int ncols = gradxArr.shape[1]
	cdef int npoints = 0

	# For most of the pixels in the image, do ... 
	pointlistval = []
	pointlistx, pointlisty = [], []

	cdef np.ndarray[np.float32_t,ndim=2] gradxxCumSum2 = np.power(gradxArr,2.).cumsum(1).cumsum(0)
	cdef np.ndarray[np.float32_t,ndim=2] gradyxCumSum2 = (gradxArr * gradyArr).cumsum(1).cumsum(0)
	cdef np.ndarray[np.float32_t,ndim=2] gradyyCumSum2 = np.power(gradyArr,2.).cumsum(1).cumsum(0)

	for y in range(bordery, nrows - bordery, nSkippedPixels + 1):
		for x in range(borderx, ncols - borderx, nSkippedPixels + 1):

			gxx = SumGradientInWindow(x,y,window_hh,window_hw,gradxxCumSum2)
			gxy = SumGradientInWindow(x,y,window_hh,window_hw,gradyxCumSum2)
			gyy = SumGradientInWindow(x,y,window_hh,window_hw,gradyyCumSum2)

			# Store the trackability of the pixel as the minimum
			# of the two eigenvalues
			val = _minEigenvalue(gxx, gxy, gyy);
			
			pointlistval.append(val)
			npoints += 1

		xRow = np.arange(borderx, ncols - borderx, nSkippedPixels + 1, np.int32)
		yRow = np.ones((len(xRow),), np.int32) * y
		#print len(xRow), len(yRow)
		pointlistx.extend(xRow)
		pointlisty.extend(yRow)

	return pointlistx,pointlisty,pointlistval

