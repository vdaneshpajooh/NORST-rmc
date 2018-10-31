This folder contains the code accompanying pre-print.

[1] "Subspace Tracking from Missing and Outlier Corrupted Data"
     Praneeth Narayanamurthy, Vahid Daneshpajooh, and Namrata Vaswani
     arXiv:1810.03051v1 [cs.LG] 6 Oct 2018

List of main files:
1. wrapper_simulated_data.m : wrapper containing the simulated data experiments
2. wrapper_VideoAnalysis.m  : wrapper for background recovery in videos.
3. NORST.m       : main function which implements the NORST algorithm for subspace tracking (on simulated data)
4. NORST_video.m : main function which implements the NORST algorithm for background recovery (real data-video)

Folders:\n
	YALL1 : folder containing files to implement ell-1 minimization.
	PROPACK : Linear Algebra toolbox for MATLAB
	data : folder containing several video data matrices for the task of background recovery

Other files:\n
	ncrpca : code implemented Non-convex Robust PCA, NIPS 14 downloaded from authors' website and its accompaniments lansvd, lanbpro etc
	cgls : fast method to implement least squares

For any further questions/suggestions please contact us @ vahidd/pkurpadn iastate edu (insert obvious symbols)
