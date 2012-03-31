#*********************************************************************
#* trackFeatures.py
#*
#*********************************************************************/

from selectGoodFeatures import KLT_verbose
from klt import *
from error import *
from convolve import *
from pyramid import *
from PIL import Image

#*********************************************************************
#* KLTTrackFeatures
#*
#* Tracks feature points from one image to the next.
#*

def KLTTrackFeatures(tc, img1, img2, featurelist):

	#_KLT_FloatImage tmpimg, floatimg1, floatimg2;
	#_KLT_Pyramid pyramid1, pyramid1_gradx, pyramid1_grady,
	#	pyramid2, pyramid2_gradx, pyramid2_grady;
	subsampling = float(tc.subsampling)
	#float xloc, yloc, xlocout, ylocout;
	#int val;
	#int indx, r;
	floatimg1_created = False
	#int i;

	assert img1.size == img2.size
	ncols, nrows = img1.size
	DEBUG_AFFINE_MAPPING = False

	if KLT_verbose >= 1:
		print "(KLT) Tracking {0} features in a {1} by {2} image...  ".format( \
			KLTCountRemainingFeatures(featurelist), ncols, nrows)

	# Check window size (and correct if necessary) 
	if tc.window_width % 2 != 1:
		tc.window_width = tc.window_width+1
		KLTWarning("Tracking context's window width must be odd.  " + \
			"Changing to {0}.".format(tc.window_width))
	
	if tc.window_height % 2 != 1:
		tc.window_height = tc.window_height+1;
		KLTWarning("Tracking context's window height must be odd.  " + \
			"Changing to {0}.".format(tc.window_height))

	if tc.window_width < 3:
		tc.window_width = 3
		KLTWarning("Tracking context's window width must be at least three.  \n" + \
			"Changing to {0}.".format(tc.window_width))
	
	if tc.window_height < 3:
		tc.window_height = 3
		KLTWarning("Tracking context's window height must be at least three.  \n" + \
			"Changing to {0}.".format(tc.window_height))
	

	# Create temporary image 
	tmpimg = Image.new("F", img1.size)

	# Process first image by converting to float, smoothing, computing 
	# pyramid, and computing gradient pyramids 
	if tc.sequentialMode and tc.pyramid_last is not None:
		pyramid1 = tc.pyramid_last
		pyramid1_gradx = tc.pyramid_last_gradx
		pyramid1_grady = tc.pyramid_last_grady
		if pyramid1.ncols[0] != ncols or pyramid1.nrows[0] != nrows:
			KLTError("(KLTTrackFeatures) Size of incoming image ({0} by {1}) " + \
			"is different from size of previous image ({2} by {3})".format( \
			ncols, nrows, pyramid1.ncols[0], pyramid1.nrows[0]))
		assert pyramid1_gradx is not None
		assert pyramid1_grady is not None
	else:
		floatimg1_created = True
		#floatimg1 = Image.new("F", img1.size)
		tmpimg = img1.convert("F")
		floatimg1 = KLTComputeSmoothedImage(tmpimg, KLTComputeSmoothSigma(tc))
		pyramid1 = _KLTCreatePyramid(ncols, nrows, int(subsampling), tc.nPyramidLevels)
		_KLTComputePyramid(floatimg1, pyramid1, tc.pyramid_sigma_fact)
		pyramid1_gradx = _KLTCreatePyramid(ncols, nrows, int(subsampling), tc.nPyramidLevels)
		pyramid1_grady = _KLTCreatePyramid(ncols, nrows, int(subsampling), tc.nPyramidLevels)
		for i in range(tc.nPyramidLevels):
			_KLTComputeGradients(pyramid1.img[i], tc.grad_sigma, 
			pyramid1_gradx.img[i],
			pyramid1_grady.img[i])

	# Do the same thing with second image
	#floatimg2 = _KLTCreateFloatImage(ncols, nrows)
	tmpimg = img2.convert("F")
	floatimg2 = KLTComputeSmoothedImage(tmpimg, KLTComputeSmoothSigma(tc))
	pyramid2 = _KLTCreatePyramid(ncols, nrows, int(subsampling), tc.nPyramidLevels)
	_KLTComputePyramid(floatimg2, pyramid2, tc.pyramid_sigma_fact)
	pyramid2_gradx = _KLTCreatePyramid(ncols, nrows, int(subsampling), tc.nPyramidLevels)
	pyramid2_grady = _KLTCreatePyramid(ncols, nrows, int(subsampling), tc.nPyramidLevels)
	for i in range(tc.nPyramidLevels):
		_KLTComputeGradients(pyramid2.img[i], tc.grad_sigma, 
		pyramid2_gradx.img[i],
		pyramid2_grady.img[i])

	# Write internal images 
	if tc.writeInternalImages:
		#char fname[80];
		for i in range(tc.nPyramidLevels):
			pyramid1.img[i].save("kltimg_tf_i{0}.pgm".format(i))
			pyramid1_gradx.img[i].save("kltimg_tf_i{0}_gx.pgm".format(i))
			pyramid1_grady.img[i].save("kltimg_tf_i{0}_gy.pgm".format(i))
			pyramid2.img[i].save("kltimg_tf_j{0}.pgm".format(i))
			pyramid2_gradx.img[i].save("kltimg_tf_j{0}_gx.pgm".format(i))
			pyramid2_grady.img[i].save("kltimg_tf_j{0}_gy.pgm".format(i))
		
	# For each feature, do ... 
	for indx, feat in enumerate(featurelist):

		# Only track features that are not lost
		if feat.val < 0: continue

		xloc = feat.x
		yloc = feat.y

		# Transform location to coarsest resolution 
		for r in range(tc.nPyramidLevels - 1, -1, -1):
			xloc /= subsampling
			yloc /= subsampling

		xlocout = xloc
		ylocout = yloc

		# Beginning with coarsest resolution, do ... 
		for r in range(tc.nPyramidLevels - 1, -1, -1):

			# Track feature at current resolution 
			xloc *= subsampling
			yloc *= subsampling
			xlocout *= subsampling
			ylocout *= subsampling

			val, xlocout, ylocout = _trackFeature(xloc, yloc, 
				xlocout, ylocout,
				pyramid1.img[r], 
				pyramid1_gradx.img[r], pyramid1_grady.img[r], 
				pyramid2.img[r], 
				pyramid2_gradx.img[r], pyramid2_grady.img[r],
				tc.window_width, tc.window_height,
				tc.step_factor,
				tc.max_iterations,
				tc.min_determinant,
				tc.min_displacement,
				tc.max_residue,
				tc.lighting_insensitive)

			if val==kltState.KLT_SMALL_DET or val==kltState.KLT_OOB:
				break

		# Record feature 
		if val == kltState.KLT_OOB:
			feat.x   = -1.0
			feat.y   = -1.0
			feat.val = kltState.KLT_OOB
			if feat.aff_img is not None: _KLTFreeFloatImage(feat.aff_img)
			if feat.aff_img_gradx is not None: _KLTFreeFloatImage(feat.aff_img_gradx)
			if feat.aff_img_grady is not None: _KLTFreeFloatImage(feat.aff_img_grady)
			feat.aff_img = None
			feat.aff_img_gradx = None
			feat.aff_img_grady = None

		elif _outOfBounds(xlocout, ylocout, ncols, nrows, tc.borderx, tc.bordery):
			feat.x   = -1.0
			feat.y   = -1.0
			feat.val = kltState.KLT_OOB
			if feat.aff_img is not None: _KLTFreeFloatImage(feat.aff_img)
			if feat.aff_img_gradx is not None: _KLTFreeFloatImage(feat.aff_img_gradx)
			if feat.aff_img_grady is not None: _KLTFreeFloatImage(feat.aff_img_grady)
			feat.aff_img = None
			feat.aff_img_gradx = None
			feat.aff_img_grady = None

		elif val == kltState.KLT_SMALL_DET:
			feat.x   = -1.0
			feat.y   = -1.0
			feat.val = kltState.KLT_SMALL_DET
			if feat.aff_img is not None: _KLTFreeFloatImage(feat.aff_img)
			if feat.aff_img_gradx is not None: _KLTFreeFloatImage(feat.aff_img_gradx)
			if feat.aff_img_grady is not None: _KLTFreeFloatImage(feat.aff_img_grady)
			feat.aff_img = None
			feat.aff_img_gradx = None
			feat.aff_img_grady = None

		elif val == kltState.KLT_LARGE_RESIDUE:
			feat.x   = -1.0
			feat.y   = -1.0
			feat.val = kltState.KLT_LARGE_RESIDUE;
			if feat.aff_img is not None: _KLTFreeFloatImage(feat.aff_img)
			if feat.aff_img_gradx is not None: _KLTFreeFloatImage(feat.aff_img_gradx)
			if feat.aff_img_grady is not None: _KLTFreeFloatImage(feat.aff_img_grady)
			feat.aff_img = None
			feat.aff_img_gradx = None
			feat.aff_img_grady = None

		elif val == kltState.KLT_MAX_ITERATIONS:
			feat.x   = -1.0
			feat.y   = -1.0
			feat.val = kltState.KLT_MAX_ITERATIONS
			if feat.aff_img is not None: _KLTFreeFloatImage(feat.aff_img)
			if feat.aff_img_gradx is not None: _KLTFreeFloatImage(feat.aff_img_gradx)
			if feat.aff_img_grady is not None: _KLTFreeFloatImage(feat.aff_img_grady)
			feat.aff_img = None
			feat.aff_img_gradx = None
			feat.aff_img_grady = None

		else:
			feat.x = xlocout;
			feat.y = ylocout;
			feat.val = kltState.KLT_TRACKED;
			if tc.affineConsistencyCheck >= 0 and val == kltState.KLT_TRACKED: #for affine mapping
				border = 2 # add border for interpolation 

				if DEBUG_AFFINE_MAPPING:
					glob_index = indx

				if feat.aff_img is None:
					#save image and gradient for each feature at finest resolution after first successful track */
					feat.aff_img = _KLTCreateFloatImage((tc.affine_window_width+border), (tc.affine_window_height+border))
					feat.aff_img_gradx = _KLTCreateFloatImage((tc.affine_window_width+border), (tc.affine_window_height+border))
					feat.aff_img_grady = _KLTCreateFloatImage((tc.affine_window_width+border), (tc.affine_window_height+border))
					_am_getSubFloatImage(pyramid1.img[0],xloc,yloc,feat.aff_img)
					_am_getSubFloatImage(pyramid1_gradx.img[0],xloc,yloc,feat.aff_img_gradx)
					_am_getSubFloatImage(pyramid1_grady.img[0],xloc,yloc,feat.aff_img_grady)
					feat.aff_x = xloc - int(xloc) + (tc.affine_window_width+border)/2
					feat.aff_y = yloc - int(yloc) + (tc.affine_window_height+border)/2
				else:
					# affine tracking 
					val, xlocout, ylocout = _am_trackFeatureAffine(feat.aff_x, feat.aff_y,
						xlocout, ylocout,
						feat.aff_img, 
						feat.aff_img_gradx, 
						feat.aff_img_grady,
						pyramid2.img[0], 
						pyramid2_gradx.img[0], pyramid2_grady.img[0],
						tc.affine_window_width, tc.affine_window_height,
						tc.step_factor,
						tc.affine_max_iterations,
						tc.min_determinant,
						tc.min_displacement,
						tc.affine_min_displacement,
						tc.affine_max_residue, 
						tc.lighting_insensitive,
						tc.affineConsistencyCheck,
						tc.affine_max_displacement_differ,
						feat.aff_Axx, #out?
						feat.aff_Ayx, #out?
						feat.aff_Axy, #out?
						feat.aff_Ayy ) #out?
					feat.val = val
					if val != kltState.KLT_TRACKED:
						feat.x   = -1.0;
						feat.y   = -1.0;
						feat.aff_x = -1.0;
						feat.aff_y = -1.0;
						# free image and gradient for lost feature
						_KLTFreeFloatImage(feat.aff_img);
						_KLTFreeFloatImage(feat.aff_img_gradx);
						_KLTFreeFloatImage(feat.aff_img_grady);
						feat.aff_img = None
						feat.aff_img_gradx = None
						feat.aff_img_grady = None
					else:
						pass
						#feat.x = xlocout;
						#feat.y = ylocout;

	if tc.sequentialMode:
		tc.pyramid_last = pyramid2
		tc.pyramid_last_gradx = pyramid2_gradx
		tc.pyramid_last_grady = pyramid2_grady

	if KLT_verbose >= 1:
		print "\n\t{0} features successfully tracked.".format(KLTCountRemainingFeatures(featurelist))
		if tc.writeInternalImages:
			print "\tWrote images to 'kltimg_tf*.pgm'."



