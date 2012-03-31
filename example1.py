#**********************************************************************
#Finds the 100 best features in an image, and tracks these
#features to the next image.  Saves the feature
#locations (before and after tracking) to text files and to PPM files, 
#and prints the features to the screen.
#**********************************************************************

#include "pnmio.h"
from klt import *
from PIL import Image
from selectGoodFeatures import *
from writeFeatures import *
from trackFeatures import *
import pickle

if __name__=="__main__":
	tc = KLT_TrackingContext()
	nFeatures = 1
	
	#KLTPrintTrackingContext(tc)

	img1 = Image.open("img0.pgm")
	img2 = Image.open("img1.pgm")
	ncols, nrows = img1.size

	fl = KLTSelectGoodFeatures(tc, img1, nFeatures)

	tc.writeInternalImages = True
	print "\nIn first image:"
	for i, feat in enumerate(fl):
		print "Feature #{0}:  ({1},{2}) with value of {3}".format(i, feat.x, feat.y, feat.val)

	KLTWriteFeatureListToPPM(fl, img1, "feat1.ppm")
	#KLTWriteFeatureList(fl, "feat1.txt", "%3d")

	#pickle.dump(tc, open("context.dat","w"))
	#pickle.dump(fl, open("featurelist.dat","w"))

	#img1 = Image.open("img0.pgm")
	#img2 = Image.open("img1.pgm")

	#tc = pickle.load(open("context.dat"))
	#tc.writeInternalImages = True
	#KLTPrintTrackingContext(tc)

	#fl = pickle.load(open("featurelist.dat"))

	KLTTrackFeatures(tc, img1, img2, fl)

	#print "\nIn second image:"
	#for i, feat in enumerate(fl):
	#	print "Feature #{0}:  ({1},{2}) with value of {3}".format(i, feat.x, feat.y, feat.val)

	KLTWriteFeatureListToPPM(fl, img2, "feat2.ppm")
	#KLTWriteFeatureList(fl, "feat2.fl", NULL)      # binary file 
	#KLTWriteFeatureList(fl, "feat2.txt", "%5.1f")  # text file   




