#*********************************************************************
#* klt.h
#*
#* Kanade-Lucas-Tomasi tracker
#*********************************************************************/

#typedef float KLT_locType;
#typedef unsigned char KLT_PixelType;

#define KLT_BOOL int

#ifndef TRUE
#define TRUE  1
#define FALSE 0
#endif

#ifndef NULL
#define NULL  0
#endif

class kltState:
	KLT_TRACKED = 0
	KLT_NOT_FOUND = -1
	KLT_SMALL_DET = -2
	KLT_MAX_ITERATIONS = -3
	KLT_OOB = -4
	KLT_LARGE_RESIDUE = -5

#include "klt_util.h" /* for affine mapping */

import math
from error import *
from convolve import *
from klt_util import *

#*******************
#* Structures
#*

class KLT_TrackingContext:
	def __init__(self):
		#Set values to default values
		self.mindist = 10
		self.window_width = 7
		self.window_height = 7
		self.sequentialMode = False
		self.retainTrackers = False
		self.smoothBeforeSelecting = True
		self.writeInternalImages = False
		self.lighting_insensitive = False
		self.min_eigenvalue = 1
		self.min_determinant = 0.01
		self.max_iterations = 10
		self.min_displacement = 0.1
		self.max_residue = None #10.0
		self.grad_sigma = 1.0
		self.smooth_sigma_fact = 0.1
		self.pyramid_sigma_fact = 0.9
		self.step_factor = 1.0
		self.nSkippedPixels = 0
		self.pyramid_last = None
		self.pyramid_last_gradx = None
		self.pyramid_last_grady = None
		# for affine mapping
		self.affineConsistencyCheck = -1
		self.affine_window_width = 15
		self.affine_window_height = 15
		self.affine_max_iterations = 10
		self.affine_max_residue = 10.
		self.affine_min_displacement = 0.02
		self.affine_max_displacement_differ = 1.5

		# Change nPyramidLevels and subsampling
		search_range=15
		self.KLTChangeTCPyramid(search_range)
	
		# Update border, which is dependent upon
		# smooth_sigma_fact, pyramid_sigma_fact, window_size, and subsampling
		self.KLTUpdateTCBorder()


	def KLTChangeTCPyramid(self,search_range):

		# Check window size (and correct if necessary) 
		if self.window_width % 2 != 1:
			self.window_width = self.window_width+1
			KLTWarning("(KLTChangeTCPyramid) Window width must be odd.  Changing to {0}\n".format(self.window_width))

		if self.window_height % 2 != 1:
			self.window_height = self.window_height+1
			KLTWarning("(KLTChangeTCPyramid) Window height must be odd.  Changing to {0}.\n".format(self.window_height))
		
		if self.window_width < 3:
			self.window_width = 3;
			KLTWarning("(KLTChangeTCPyramid) Window width must be at least three.  \nChanging to {0}.\n".format(self.window_width))

		if self.window_height < 3:
			self.window_height = 3
			KLTWarning("(KLTChangeTCPyramid) Window height must be at least three.  \nChanging to %d.\n".format(self.window_height))
		
		window_halfwidth = min(self.window_width,self.window_height)/2.0

		subsampling = float (search_range) / window_halfwidth

		if subsampling < 1.0:		# 1.0 = 0+1
			self.nPyramidLevels = 1
		elif subsampling <= 3.0:	# 3.0 = 2+1
			self.nPyramidLevels = 2
			self.subsampling = 2
		elif subsampling <= 5.0:  	# 5.0 = 4+1
			self.nPyramidLevels = 2
			self.subsampling = 4
		elif subsampling <= 9.0:  	# 9.0 = 8+1
			self.nPyramidLevels = 2
			self.subsampling = 8
		else:
			# The following lines are derived from the formula:
			#   search_range = 
			#   window_halfwidth * \sum_{i=0}^{nPyramidLevels-1} 8^i,
			#   which is the same as:
			#   search_range = 
			#   window_halfwidth * (8^nPyramidLevels - 1)/(8 - 1).
			#   Then, the value is rounded up to the nearest integer. 
			val = float (math.log(7.0*subsampling+1.0)/math.log(8.0))
			self.nPyramidLevels = int(val + 0.99)
		  	self.subsampling = 8
		


	#*********************************************************************
	#* Updates border, which is dependent upon 
	#* smooth_sigma_fact, pyramid_sigma_fact, window_size, and subsampling
	#*

	def KLTUpdateTCBorder(self):
	
		num_levels = self.nPyramidLevels;
		ss = self.subsampling;

		# Check window size (and correct if necessary)
		if self.window_width % 2 != 1:
			self.window_width = self.window_width+1
			KLTWarning("(KLTUpdateTCBorder) Window width must be odd.  Changing to {0}.\n".format(self.window_width))
		
		if self.window_height % 2 != 1:
			self.window_height = self.window_height+1
			KLTWarning("(KLTUpdateTCBorder) Window height must be odd.  Changing to {0}.\n".format(self.window_height))
		
		if self.window_width < 3:
			self.window_width = 3
			KLTWarning("(KLTUpdateTCBorder) Window width must be at least three.  \nChanging to {0}.\n".format(self.window_width))
		
		if self.window_height < 3:
			self.window_height = 3
			KLTWarning("(KLTUpdateTCBorder) Window height must be at least three.  \nChanging to {0}.\n".format(self.window_height))
		
		window_hw = max(self.window_width, self.window_height)/2

		# Find widths of convolution windows
		gauss_width, gaussderiv_width = KLTGetKernelWidths(KLTComputeSmoothSigma(self))

		smooth_gauss_hw = gauss_width/2
		gauss_width, gaussderiv_width = KLTGetKernelWidths(_pyramidSigma(self))
		pyramid_gauss_hw = gauss_width/2

		# Compute the # of invalid pixels at each level of the pyramid.
		#   n_invalid_pixels is computed with respect to the ith level   
		#   of the pyramid.  So, e.g., if n_invalid_pixels = 5 after   
		#   the first iteration, then there are 5 invalid pixels in   
		#   level 1, which translated means 5*subsampling invalid pixels   
		#   in the original level 0. 
		n_invalid_pixels = smooth_gauss_hw
		for i in range(1,num_levels):
			val = (float(n_invalid_pixels) + pyramid_gauss_hw) / ss
			n_invalid_pixels = int(val + 0.99)  # Round up
	
		# ss_power = ss^(num_levels-1) 
		ss_power = 1
		for i in range(1,num_levels):
			ss_power *= ss

		# Compute border by translating invalid pixels back into
		# original image
		border = (n_invalid_pixels + window_hw) * ss_power

		self.borderx = border
		self.bordery = border
	

#*********************************************************************
#* NOTE:  Manually must ensure consistency with _KLTComputePyramid()
#*
 
def _pyramidSigma(tc):
	return (tc.pyramid_sigma_fact * tc.subsampling)


  #Available to user
  #int mindist		# min distance b/w features
  #int window_width, window_height
  #KLT_BOOL sequentialMode;	/* whether to save most recent image to save time */
  #/* can set to TRUE manually, but don't set to */
  #/* FALSE manually */
  #KLT_BOOL smoothBeforeSelecting;	/* whether to smooth image before */
  #/* selecting features */
  #KLT_BOOL writeInternalImages;	/* whether to write internal images */
  #/* tracking features */
  #KLT_BOOL lighting_insensitive;  /* whether to normalize for gain and bias (not in original algorithm) */
  
  #/* Available, but hopefully can ignore */
  #int min_eigenvalue;		/* smallest eigenvalue allowed for selecting */
  #float min_determinant;	/* th for determining lost */
  #float min_displacement;	/* th for stopping tracking when pixel changes little */
  #int max_iterations;		/* th for stopping tracking when too many iterations */
  #float max_residue;		/* th for stopping tracking when residue is large */
  #float grad_sigma;
  #float smooth_sigma_fact;
  #float pyramid_sigma_fact;
  #float step_factor;  /* size of Newton steps; 2.0 comes from equations, 1.0 seems to avoid overshooting */
  #int nSkippedPixels;		/* # of pixels skipped when finding features */
  #int borderx;			/* border in which features will not be found */
  #int bordery;
  #int nPyramidLevels;		/* computed from search_ranges */
  #int subsampling;		/* 		" */

  
  # for affine mapping 
  #int affine_window_width, affine_window_height;
  #int affineConsistencyCheck; /* whether to evaluates the consistency of features with affine mapping 
  #                            -1 = don't evaluates the consistency
  #                            0 = evaluates the consistency of features with translation mapping
  #                            1 = evaluates the consistency of features with similarity mapping
  #                            2 = evaluates the consistency of features with affine mapping
  
  #int affine_max_iterations;  
  #float affine_max_residue;
  #float affine_min_displacement;        
  #float affine_max_displacement_differ; /* th for the difference between the displacement calculated 
  #by the affine tracker and the frame to frame tracker in pel*/

  # User must not touch these
  #void *pyramid_last;
  #void *pyramid_last_gradx;
  #void *pyramid_last_grady;
#KLT_TrackingContextRec, *KLT_TrackingContext;

class KLT_Feature:
	def __init__(self):
		x = None
		y = None
		val = None
		# for affine mapping
		aff_img = None
		aff_img_gradx = None
		aff_img_grady = None
		aff_x = None
		aff_y = None
		aff_Axx = None
		aff_Ayx = None
		aff_Axy = None
		aff_Ayy = None
# KLT_FeatureRec, *KLT_Feature;

#class KLT_FeatureList:
#	def __init__(self, nFeatures):
#		#self.nFeatures = nFeatures
#		self.feature = [KLT_Feature() for i in range(nFeatures)]
#KLT_FeatureListRec, *KLT_FeatureList;

class KLT_FeatureHistory:
  pass
  #int nFrames;
  #KLT_Feature *feature;
#KLT_FeatureHistoryRec, *KLT_FeatureHistory;

class KLT_FeatureTable:
  pass
  #int nFrames;
  #int nFeatures;
  #KLT_Feature **feature;
#KLT_FeatureTableRec, *KLT_FeatureTable;

def KLTPrintTrackingContext(tc):

	print tc
	print "\n\nTracking context:\n"
	print "\tmindist = {0}".format(tc.mindist)
	print "\twindow_width = {0}".format(tc.window_width)
	print "\twindow_height = {0}".format(tc.window_height)
	print "\tsequentialMode = {0}".format(tc.sequentialMode)
	print "\tsmoothBeforeSelecting = {0}".format(tc.smoothBeforeSelecting)
	print "\twriteInternalImages = {0}".format(tc.writeInternalImages)

	print "\tmin_eigenvalue = {0}".format(tc.min_eigenvalue)
	print "\tmin_determinant = {0}".format(tc.min_determinant)
	print "\tmin_displacement = {0}".format(tc.min_displacement)
	print "\tmax_iterations = {0}".format(tc.max_iterations)
	print "\tmax_residue = {0}".format(tc.max_residue)
	print "\tgrad_sigma = {0}".format(tc.grad_sigma)
	print "\tsmooth_sigma_fact = {0}".format(tc.smooth_sigma_fact)
	print "\tpyramid_sigma_fact = {0}".format(tc.pyramid_sigma_fact)
	print "\tnSkippedPixels = {0}".format(tc.nSkippedPixels)
	print "\tborderx = {0}".format(tc.borderx)
	print "\tbordery = {0}".format(tc.bordery)
	print "\tnPyramidLevels = {0}".format(tc.nPyramidLevels)
	print "\tsubsampling = {0}".format(tc.subsampling)

	print "\n\tpyramid_last = {0}".format(tc.pyramid_last)
	print "\tpyramid_last_gradx = {0}".format(tc.pyramid_last_gradx)
	print "\tpyramid_last_grady = {0}".format(tc.pyramid_last_grady)
	print "\n"

#*********************************************************************
#* KLTCountRemainingFeatures
#*

def KLTCountRemainingFeatures(fl):

	count = 0
	for feat in fl:
		if feat.val >= 0:
			count = count + 1
	return count


