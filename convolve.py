
import math

class ConvolutionKernel:
	def __init__(self, maxKernelWidth = 71):
		self.width = None
		self.data = [0. for i in range(maxKernelWidth)]

#*********************************************************************
#* _computeKernels
#*

def _computeKernels(sigma):
	maxKernelWidth = 71
	gauss = ConvolutionKernel(maxKernelWidth)
	gaussderiv = ConvolutionKernel(maxKernelWidth)

	factor = 0.01   # for truncating tail

	assert maxKernelWidth % 2 == 1
	assert sigma >= 0.0

	# Compute kernels, and automatically determine widths */

	hw = maxKernelWidth / 2
	max_gauss = 1.0
	max_gaussderiv = float(sigma*math.exp(-0.5))
	
	# Compute gauss and deriv 
	for i in range(-hw,hw+1):
		gauss.data[i+hw] = float (math.exp(-i*i / (2*sigma*sigma)))
		gaussderiv.data[i+hw] = -i * gauss.data[i+hw]

    	# Compute widths
	gauss.width = maxKernelWidth;
	for i in range(-hw, abs(gauss.data[i+hw] / max_gauss) < factor):
		gauss.width -= 2
	gaussderiv.width = maxKernelWidth

	for i in range(-hw, abs(gaussderiv.data[i+hw] / max_gaussderiv) < factor) :
		gaussderiv.width -= 2
	if gauss.width == maxKernelWidth or gaussderiv.width == maxKernelWidth:
		KLTError("(_computeKernels) maxKernelWidth {0} is too small for a sigma of {1}".format(maxKernelWidth, sigma))

	# Shift if width less than maxKernelWidth 
	for i in range(gauss.width):
		gauss.data[i] = gauss.data[i+(maxKernelWidth-gauss.width)/2]
	for i in range(gaussderiv.width):
		gaussderiv.data[i] = gaussderiv.data[i+(maxKernelWidth-gaussderiv.width)/2]
	# Normalize gauss and deriv 
	hw = gaussderiv.width / 2
	den = 0.0;
	for i in range(gauss.width): 
		den += gauss.data[i]
	for i in range(gauss.width): 
		gauss.data[i] /= den
	den = 0.0
	for i in range(-hw,hw+1): 
		den -= i*gaussderiv.data[i+hw]
	for i in range(-hw,hw+1): 
		gaussderiv.data[i+hw] /= den

	sigma_last = sigma

	return gauss, gaussderiv

#*********************************************************************
#* KLTGetKernelWidths
#*
#*

def KLTGetKernelWidths(sigma):
	gauss_kernel, gaussderiv_kernel = _computeKernels(sigma)
	return gauss_kernel.width, gaussderiv_kernel.width


