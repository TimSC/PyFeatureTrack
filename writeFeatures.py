
from selectGoodFeatures import KLT_verbose
from klt import *

#*********************************************************************
#* KLTWriteFeatureListToPPM
#*

def KLTWriteFeatureListToPPM(featurelist, greyimg, filename):

	#int nbytes = ncols * nrows * sizeof(char);
	#uchar *redimg, *grnimg, *bluimg;
	#int offset;
	#int x, y, xx, yy;
	#int i;
	ncols, nrows = greyimg.size

	if KLT_verbose:
		print "(KLT) Writing {0} features to PPM file: '{1}'".format(KLTCountRemainingFeatures(featurelist), filename)

	tmp = greyimg.copy()
	tmp = tmp.convert("RGB")
	tmpl = tmp.load()

	# Overlay features in red 
	for feat in featurelist:
		if feat.val >= 0:
			x = int(feat.x + 0.5);
			y = int(feat.y + 0.5);
			for yy in range(y - 1,y + 2):
				for xx in range(x - 1,x + 2):
					if xx >= 0 and yy >= 0 and xx < ncols and yy < nrows:
						tmpl[xx,yy] = (255,0,0)
	
	# Write to PPM file 
	tmp.save(filename)

