
from __future__ import print_function
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
		print("(KLT) Writing {0} features to PPM file: '{1}'".format(KLTCountRemainingFeatures(featurelist), filename))

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


#*********************************************************************
#* KLTWriteFeatureList()
#* KLTWriteFeatureHistory()
#* KLTWriteFeatureTable()
#* 
#* Writes features to file or to screen.
#*
#* INPUTS
#* fname: name of file to write data; if NULL, then print to stderr
#* fmt:   format for printing (e.g., "%5.1f" or "%3d");
#*		if NULL, and if fname is not NULL, then write to binary file.
#*

def KLTWriteFeatureList(fl, fname, fmt):

	#FILE *fp;
	#char format[100];
	#char type;
	#int i;

	fmtStr = "binary" if fmt is None else "text"
	if KLT_verbose >= 1 and fname is not None:
		print("(KLT) Writing feature list to {0} file: '{1}'".format(fmtStr, fname))

	if fmt is not None: # text file or stderr
		fp = _printSetupTxt(fname, fmt, format, type)
		_printHeader(fp, format, FEATURE_LIST, 0, fl.nFeatures)
	
	  	#for (i = 0 ; i < fl->nFeatures ; i++):
		#	fprintf(fp, "%7d | ", i)
		#	_printFeatureTxt(fp, fl.feature[i], format, type)
		#	fprintf(fp, "\n")

		_printShutdown(fp)

	else: # binary file 
		fp = _printSetupBin(fname)
		#fwrite(binheader_fl, sizeof(char), BINHEADERLENGTH, fp);
		#fwrite(&(fl.nFeatures), sizeof(int), 1, fp)
		#for (i = 0 ; i < fl.nFeatures ; i++):
		#	_printFeatureBin(fp, fl.feature[i])
	
		fclose(fp)
	

