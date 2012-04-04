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

def _minEigenvalue(gxx, gxy, gyy):
	return float((gxx + gyy - math.sqrt((gxx - gyy)*(gxx - gyy) + 4.*gxy*gxy))/2.)

#*********************************************************************

def SumGradientInWindow(x,y,window_hh,window_hw,gradxxCumSum):

	#Sum the gradients in the surrounding window with numpy
	gxx = (gradxxCumSum[y-window_hh: y+window_hh+1,x+window_hw] - gradxxCumSum[y-window_hh: y+window_hh+1, x-window_hw-1]).sum()
	return gxx

#**********************************************************************

def ScanImageForGoodFeatures(gradxArr, gradyArr, borderx, bordery, window_hw, window_hh, nSkippedPixels):
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

