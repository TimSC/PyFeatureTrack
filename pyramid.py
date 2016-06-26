#*********************************************************************
#* pyramid.py
#*
#*********************************************************************

from error import *
from convolve import *
import numpy as np

#*********************************************************************
#*
#*

class KLTPyramid:
	def __init__(self, ncols, nrows, subsampling, nlevels):

		if subsampling != 2 and subsampling != 4 and \
			subsampling != 8 and subsampling != 16 and subsampling != 32:
			KLTError("(_KLTCreatePyramid)  Pyramid's subsampling must " + \
			   "be either 2, 4, 8, 16, or 32")

		# Set parameters
		self.subsampling = subsampling;
		self.nLevels = nlevels;
		self.img = []
		self.ncols = []
		self.nrows = []

		# Allocate memory for each level of pyramid and assign pointers
		for i in range(nlevels):
			self.img.append(None)
			self.ncols.append(ncols)  
			self.nrows.append(nrows)
			ncols /= subsampling  
			nrows /= subsampling

	def Compute(self, img, sigma_fact):

		nrows = img.shape[0]
		ncols = img.shape[1]
		subsampling = self.subsampling
		subhalf = subsampling / 2
		sigma = subsampling * sigma_fact;  # empirically determined
		#int oldncols;
		#int i, x, y;
	
		if subsampling != 2 and subsampling != 4 and \
			subsampling != 8 and subsampling != 16 and subsampling != 32:
			KLTError("(_KLTComputePyramid)  Pyramid's subsampling must " + \
			   "be either 2, 4, 8, 16, or 32")

		assert self.ncols[0] == ncols
		assert self.nrows[0] == nrows

		# Copy original image to level 0 of pyramid
		self.img[0] = img

		currimg = img
		for i in range(1,self.nLevels):
			tmpimg = KLTComputeSmoothedImage(currimg, sigma)
			#tmpimgL = tmpimg.load()

			# Subsample
			oldncols = ncols
			ncols = int(ncols / subsampling)
			nrows = int(nrows / subsampling)
			#subsampImg = Image.new("F",(ncols,nrows))
			subsampImg = np.empty((nrows,ncols), np.float32)
			#subsampImgL = subsampImg.load()
			for y in range(nrows):
				for x in range(ncols):
					subsampImg[y,x] = tmpimg[int(subsampling*y+subhalf), int(subsampling*x+subhalf)]

			self.img[i] = subsampImg

			# Reassign current image 
			currimg = self.img[i]
				

 



