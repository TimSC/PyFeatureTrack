# cython: profile=True
import math, numpy as np

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

cdef float SumGradientInWindow(int x,int y,int window_hh,int window_hw,gradxxCumSum):

	cdef float total

	#Sum the gradients in the surrounding window with numpy
	higher = gradxxCumSum[y-window_hh: y+window_hh+1,x+window_hw] 
	lower = gradxxCumSum[y-window_hh: y+window_hh+1, x-window_hw-1]

	total = (higher - lower).sum()
	return total

#**********************************************************************

def ScanImageForGoodFeatures(gradxArr, gradyArr, borderx, bordery, window_hw, window_hh, nSkippedPixels):
	cdef float val, gxx, gxy, gyy
	cdef int x, y

	# For most of the pixels in the image, do ... 
	pointlistx,pointlisty,pointlistval = [], [], []
	npoints = 0
	nrows, ncols = gradxArr.shape
	
	gradxxCumSum = np.power(gradxArr,2.).cumsum(axis=1)
	gradyxCumSum = (gradxArr * gradyArr).cumsum(axis=1)
	gradyyCumSum = np.power(gradyArr,2.).cumsum(axis=1)

	for y in range(bordery, nrows - bordery):
		for x in range(borderx, ncols - borderx):

			gxx = SumGradientInWindow(x,y,window_hh,window_hw,gradxxCumSum)
			gxy = SumGradientInWindow(x,y,window_hh,window_hw,gradyxCumSum)
			gyy = SumGradientInWindow(x,y,window_hh,window_hw,gradyyCumSum)

			# Store the trackability of the pixel as the minimum
			# of the two eigenvalues 
			pointlistx.append(x)
			pointlisty.append(y)
			val = _minEigenvalue(gxx, gxy, gyy);
			
			pointlistval.append(int(val))
			npoints = npoints + 1
			x += nSkippedPixels + 1

		y += nSkippedPixels + 1

	return pointlistx,pointlisty,pointlistval

