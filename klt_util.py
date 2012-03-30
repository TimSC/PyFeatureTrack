

def KLTComputeSmoothSigma(tc):
	return (tc.smooth_sigma_fact * max(tc.window_width, tc.window_height))

