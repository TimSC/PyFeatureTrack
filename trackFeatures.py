#*********************************************************************
#* trackFeatures.py
#*
#*********************************************************************/

from selectGoodFeatures import KLT_verbose
from klt import *
from error import *
from convolve import *
from pyramid import *
from klt_util import *
from PIL import Image
import trackFeaturesUtils


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

#*********************************************************************
#* _sumAbsFloatWindow
#*

def _sumAbsFloatWindow(fw, width, height):
	sum = 0.
	fwind = 0

	for h in range(height,0,-1):
		for w in range(width):
			sum += float(abs(fw[fwind]))
			fwind += 1

	return sum

#*********************************************************************
#* _trackFeature
#*
#* Tracks a feature point from one image to the next.
#*
#* RETURNS
#* KLT_SMALL_DET if feature is lost,
#* KLT_MAX_ITERATIONS if tracking stopped because iterations timed out,
#* KLT_TRACKED otherwise.
#*

def _trackFeature(
	x1,  # location of window in first image 
	y1,
	x2, # starting location of search in second image
	y2,
	img1, 
	gradx1,
	grady1,
	img2, 
	gradx2,
	grady2,
	width,           # size of window
	height,
	step_factor, # 2.0 comes from equations, 1.0 seems to avoid overshooting
	max_iterations,
	small,         # determinant threshold for declaring KLT_SMALL_DET 
	th,            # displacement threshold for stopping               
	max_residue,   # residue threshold for declaring KLT_LARGE_RESIDUE 
	lighting_insensitive): # whether to normalize for gain and bias 

	#_FloatWindow imgdiff, gradx, grady;
	#float gxx, gxy, gyy, ex, ey, dx, dy;
	iteration = 0
	hw = width/2
	hh = height/2
	nc = img1.shape[1]
	nr = img1.shape[0]
	one_plus_eps = 1.001   # To prevent rounding errors 

	# Allocate memory for windows
	#imgdiff = [0. for i in range(width, height)]
	#gradx   = [0. for i in range(width, height)]
	#grady   = [0. for i in range(width, height)]

	# Iteratively update the window position 
	while True:

		# If out of bounds, exit loop 
		if x1-hw < 0. or nc-( x1+hw) < one_plus_eps or \
			x2-hw < 0. or nc-(x2+hw) < one_plus_eps or \
			y1-hh < 0. or nr-( y1+hh) < one_plus_eps or \
			y2-hh < 0. or nr-(y2+hh) < one_plus_eps:
			status = kltState.KLT_OOB
			break

		# Compute gradient and difference windows 
		if lighting_insensitive:
			imgdiff = trackFeaturesUtils._computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, x2, y2, width, height)
			gradx, grady = trackFeaturesUtils._computeGradientSumLightingInsensitive(gradx1, grady1, gradx, grady2, img1, img2, x1, y1, x2, y2, width, height)
		else:
			imgdiff = trackFeaturesUtils._computeIntensityDifference(img1, img2, x1, y1, x2, y2, width, height)
			gradx, grady = trackFeaturesUtils._computeGradientSum(gradx1, grady1, gradx2, grady2, x1, y1, x2, y2, width, height)

		# Use these windows to construct matrices 
		gxx, gxy, gyy = _compute2by2GradientMatrix(gradx, grady, width, height)
		ex, ey = _compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor)

		# Using matrices, solve equation for new displacement */
		status, dx, dy = _solveEquation(gxx, gxy, gyy, ex, ey, small)
		if status == kltState.KLT_SMALL_DET: break

		x2 += dx
		y2 += dy
		iteration += 1

		if not ((abs(dx)>=th or abs(dy)>=th) and iteration < max_iterations): break

	# Check whether window is out of bounds 
	if x2-hw < 0.0 or nc-(x2+hw) < one_plus_eps or y2-hh < 0.0 or nr-(y2+hh) < one_plus_eps:
		status = kltState.KLT_OOB

	# Check whether residue is too large 
	if status == kltState.KLT_TRACKED:
		if lighting_insensitive:
			imgdiff = trackFeaturesUtils._computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, x2, y2, width, height)
	  	else:
			imgdiff = trackFeaturesUtils._computeIntensityDifference(img1, img2, x1, y1, x2, y2, width, height)

		if _sumAbsFloatWindow(imgdiff, width, height)/(width*height) > max_residue:
			status = kltState.KLT_LARGE_RESIDUE

	# Return appropriate value 
	if status == kltState.KLT_SMALL_DET: return kltState.KLT_SMALL_DET, x2, y2
	elif status == kltState.KLT_OOB: return kltState.KLT_OOB, x2, y2
	elif status == kltState.KLT_LARGE_RESIDUE: return kltState.KLT_LARGE_RESIDUE, x2, y2
	elif iteration >= max_iterations: return kltState.KLT_MAX_ITERATIONS, x2, y2
	else: return kltState.KLT_TRACKED, x2, y2




#*********************************************************************

def _outOfBounds(x, y, ncols, nrows, borderx, bordery):
	return x < borderx or x > ncols-1-borderx or y < bordery or y > nrows-1-bordery

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
		tmpimg = np.array(img1.convert("F"))
		floatimg1 = KLTComputeSmoothedImage(tmpimg, KLTComputeSmoothSigma(tc))
		pyramid1 = KLTPyramid(ncols, nrows, int(subsampling), tc.nPyramidLevels)
		pyramid1.Compute(floatimg1, tc.pyramid_sigma_fact)
		pyramid1_gradx = KLTPyramid(ncols, nrows, int(subsampling), tc.nPyramidLevels)
		pyramid1_grady = KLTPyramid(ncols, nrows, int(subsampling), tc.nPyramidLevels)
		for i in range(tc.nPyramidLevels):
			pyramid1_gradx.img[i],pyramid1_grady.img[i] = KLTComputeGradients(pyramid1.img[i], tc.grad_sigma)

	# Do the same thing with second image
	#floatimg2 = _KLTCreateFloatImage(ncols, nrows)
	tmpimg = np.array(img2.convert("F"))
	floatimg2 = KLTComputeSmoothedImage(tmpimg, KLTComputeSmoothSigma(tc))
	pyramid2 = KLTPyramid(ncols, nrows, int(subsampling), tc.nPyramidLevels)
	pyramid2.Compute(floatimg2, tc.pyramid_sigma_fact)
	pyramid2_gradx = KLTPyramid(ncols, nrows, int(subsampling), tc.nPyramidLevels)
	pyramid2_grady = KLTPyramid(ncols, nrows, int(subsampling), tc.nPyramidLevels)
	for i in range(tc.nPyramidLevels):
		pyramid2_gradx.img[i], pyramid2_grady.img[i] = KLTComputeGradients(pyramid2.img[i], tc.grad_sigma)

	# Write internal images 
	if tc.writeInternalImages:
		#char fname[80];
		for i in range(tc.nPyramidLevels):
			KLTWriteFloatImageToPGM(pyramid1.img[i],"kltimg_tf_i{0}.pgm".format(i))
			KLTWriteFloatImageToPGM(pyramid1_gradx.img[i],"kltimg_tf_i{0}_gx.pgm".format(i))
			KLTWriteFloatImageToPGM(pyramid1_grady.img[i],"kltimg_tf_i{0}_gy.pgm".format(i))
			KLTWriteFloatImageToPGM(pyramid2.img[i],"kltimg_tf_j{0}.pgm".format(i))
			KLTWriteFloatImageToPGM(pyramid2_gradx.img[i],"kltimg_tf_j{0}_gx.pgm".format(i))
			KLTWriteFloatImageToPGM(pyramid2_grady.img[i],"kltimg_tf_j{0}_gy.pgm".format(i))
		
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
					#save image and gradient for each feature at finest resolution after first successful track
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



