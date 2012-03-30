#**********************************************************************
#Finds the 100 best features in an image, and tracks these
#features to the next image.  Saves the feature
#locations (before and after tracking) to text files and to PPM files, 
#and prints the features to the screen.
#**********************************************************************

#include "pnmio.h"
from klt import *

if __name__=="__main__":

	tc = KLT_TrackingContext()
	fl = KLT_FeatureList()
	nFeatures = 100
	#ncols, nrows

	KLTPrintTrackingContext(tc)
	#fl = KLTCreateFeatureList(nFeatures)

	#img1 = pgmReadFile("img0.pgm", NULL, &ncols, &nrows)
	#img2 = pgmReadFile("img1.pgm", NULL, &ncols, &nrows)

	#KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl)

	#printf("\nIn first image:\n")
	#for (i = 0 ; i < fl->nFeatures ; i++):
	#	printf("Feature #%d:  (%f,%f) with value of %d\n",
        #	i, fl->feature[i]->x, fl->feature[i]->y,
        #	fl->feature[i]->val)
  
	#KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "feat1.ppm")
	#KLTWriteFeatureList(fl, "feat1.txt", "%3d")

	#KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl)

	#printf("\nIn second image:\n");
	#for (i = 0 ; i < fl->nFeatures ; i++)
	#	printf("Feature #%d:  (%f,%f) with value of %d\n",
	#	i, fl->feature[i]->x, fl->feature[i]->y,
	#	fl->feature[i]->val)

	#KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, "feat2.ppm")
	#KLTWriteFeatureList(fl, "feat2.fl", NULL)      # binary file 
	#KLTWriteFeatureList(fl, "feat2.txt", "%5.1f")  # text file   

	#return 0


