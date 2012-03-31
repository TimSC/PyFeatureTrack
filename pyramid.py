#*********************************************************************
#* pyramid.py
#*
#*********************************************************************

from error import *
from convolve import *

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

		ncols, nrows = img.size
		subsampling = self.subsampling
		subhalf = subsampling / 2
		sigma = subsampling * sigma_fact;  # empirically determined
		#int oldncols;
		#int i, x, y;
	
		if subsampling != 2 and subsampling != 4 and \
			subsampling != 8 and subsampling != 16 and subsampling != 32:
			KLTError("(_KLTComputePyramid)  Pyramid's subsampling must " + \
			   "be either 2, 4, 8, 16, or 32")

		assert self.ncols[0] == img.size[0]
		assert self.nrows[0] == img.size[1]

		# Copy original image to level 0 of pyramid
		self.img[0] = img

		currimg = img
		for i in range(1,self.nLevels):
			tmpimg = KLTComputeSmoothedImage(currimg, sigma)
			tmpimgL = tmpimg.load()

			# Subsample
			oldncols = ncols
			ncols /= subsampling
			nrows /= subsampling
			for y in range(nrows):
				for x in range(ncols):
					subsampImg = Image.new("F",(ncols,nrows))
					subsampImgL = subsampImg.load()
					subsampImgL[x,y] = tmpimgL[subsampling*x+subhalf, subsampling*y+subhalf]
						#tmpimg->data[(subsampling*y+subhalf)*oldncols + (subsampling*x+subhalf)]
					self.img[i] = subsampImg

			# Reassign current image 
			currimg = self.img[i]
				

 



