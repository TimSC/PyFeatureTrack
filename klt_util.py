
from PIL import Image

def KLTComputeSmoothSigma(tc):
	return (tc.smooth_sigma_fact * max(tc.window_width, tc.window_height))

def KLTWriteFloatImageToPGM(img, filename):

	npixs = img.size[0] * img.size[1]
	mmax = -999999.9
	mmin = 999999.9
	imgl = img.load()
	#float fact;
	#float *ptr;
	#uchar *byteimg, *ptrout;
	#int i;

	# Calculate minimum and maximum values of float image
	mmin, mmax = img.getextrema()
	
	# Allocate memory to hold converted image
	byteimg = Image.new("L", img.size)
	byteimgl = byteimg.load()

	# Convert image from float to uchar
	if mmax != mmin:
		fact = 255.0 / (mmax-mmin)
	else:
		fact = 1.
	for i in range(img.size[0]):
		for j in range(img.size[1]):
			byteimgl[i,j] = int((imgl[i,j] - mmin) * fact)
	
	# Write uchar image to PGM
	byteimg.save(filename)


