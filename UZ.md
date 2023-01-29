# **Machine perception**

# Image formation
- Aperture - hole in camera
- Perspective projection equation: $\dfrac{f}{-y'} = \dfrac{-z}{y}$
  - $f$ ... focal length
  - $y$ ... object height
  - $y'$ ... object height on film
  - $z$ ... distance from object to camera
- Aperture size
  - Too large or too small $\implies$ blurred image
- Lens - used to focus the light to film
- Depth of field - distance between image planes at which the blurred effect is sufficiently small
  - Small aperture $\implies$ bigger DOF
- Field of view - angular measure of space percieved by the camera
  - $\phi = tan^{-1}(\dfrac{d}{2f})$
  - FOV = $2 \phi$
  - Larger $f$ $\implies$ smaller fow
- Chromatic abberation - different wave lengths refract at different angles and focus at slightly different distances
- Spherical abberation - spherical lenses do not focus the light perfectly; rays close to the edge focus closer than those at the center
- Vignetting
- Radial distortion
- Digital image - discretize image into pixels; quantize light into intensity levels
- Visible light cams - photos cause charge on each sensor cell
  - CCD
    - Reads out charge serially and digitizes
    - Better images
  - CMOS
    - Performs digitization on each cell seperately
    - Cheaper

# Image processing 1
## Image thresholding
- Image $\implies$ binary mask
- Single threshold: $F_T[i, j] = \begin{cases}
    1, \text{if} \ F[i, j] \le T \\
    0, \text{otherwise}
\end{cases}$
- Two thresholds: $F_T[i, j] = \begin{cases}
    1, \text{if} \ T_1 \le F[i, j] \le T_2 \\
    0, \text{otherwise}
\end{cases}$
- Otsu's algorithm
  - Seperate pixels into two groups by intensity threshold T
  - For each group get an average intensity and calculate $\sigma^2_{between}$
  - Select $T^*$ that maximizes variance
- Local binarization (Niblack)
  - Estimate a local threshold in a neighbourhood $W$: $T_W = \mu_W + k \sigma_W$, $k \in [-1, 1]$ set by user
  - Calculate the threshold seperately for every pixel

## Cleaning the image
- Structuring element
  - Fit - all "1" in SE cover "1" in image
  - Hit - any "1" in SE cover "1" in image
- Erosion - if $s$ fits $f$ then 1 , else 0
- Dilation - if $s$ hits $f$ then 1 , else 0
- Opening - erosion and then dilation
  - Removes structures
- Closing - dilation and then erosion
  - Fils holes

## Labelling regions
- Sequential connected components
  ```py
  if pixel == 1:
    neighbours = left and top
    if only one neighbour == 1:
      label(pixel) = label(neighbour)
    else if both neighbours == 1 && label(neighbour[left]) == label(neighbour[top]):
      label(pixel) = label(neighbours)
    else if both neighbours == 1 && label(neighbours[left]) != label(neighbours[top]):
      label(pixel) = label(neighbours[left])
      update_table_of_equivalent_labels()
    else:
      label(pixel) = form_new_label()
  ```

## Region descriptors
- Properties of connected components
  - Area
  - Perimeter
  - Compactness
  - Centroid
  - Major and minor axes
  - ...
- Ideal descriptor:
  - maps two images of the same object close-by in feature space
  - maps two images of different object far away from each other

# Image processing 2
## Color
- Aditive model - RGB
- Subtractive model  CMYK
- Color space
  - Defined by primary colors
  - A new color is a weighted sum of primaries
- Linear color spaces: CIE XYZ, RGB
- Nonlinear color space: HSV
- Uniform color space: CIE u'v'

## Color description by using histograms
- Histograms record the frequency of intensity levels
- Intensity normalization
  - Multiplying a color by a scalar changes the intensity but not the hue
  - Intensity is defined as $I = R + G + B$
  - Chromatic representation: $r = \dfrac{R}{R + G + B}$
- Calculate image similarity by comparing their histograms with a distance measure (L2, Chi-squared, Hellinger, ...)

## Filtering
- Types of image noise
  - Salt and pepper
  - Impulse noise
  - Gaussian noise
- Gaussian noise removal
  - Average pixels in a window
- Correlation filtering
  - Weighted sum of pixels in a window
  - $G[i, j] = \sum_{u = -k}^{k} \sum_{v = -k}^{k}H[u, v]F[i + u, j + v]$
- Convolution
  -  Flip the kernel
  -  $G[i, j] = \sum_{u = -k}^{k} \sum_{v = -k}^{k}H[u, v]F[i - u, j - v]$
- For a symmetric filter, correlation = convolution
- Properties of convolution
  - Shift invariant
  - Linear
  - Commutative
  - Associative
  - Identity $f * e = f$, where $e$ is a unit impulse
  - Derivative $\dfrac{\partial}{\partial x}(f * g) = (\dfrac{\partial}{\partial x}f) * g = (\dfrac{\partial}{\partial x}g) * f$
  - Boundaries
    - Crop
    - Bend image around
    - Replicate edges
    - Mirror
- Smoothing by a Gaussian
  - Convolve with a Gaussian filter
  - Variance determines the extent of smoothing
- Sharpening filter
  - $\begin{bmatrix}
    0 & 0 & 0 \\
    0 & 2 & 0 \\
    0 & 0 & 0
  \end{bmatrix} - \dfrac{1}{9} \begin{bmatrix}
    1 & 1 & 1 \\
    1 & 1 & 1 \\
    1 & 1 & 1
  \end{bmatrix}$
- Median filter
  - Replace the pixel intensity with the median of intensities within a small patch
  - Good for salt and pepper noise

## Linear filtering as template matching
- Apply correlation with a template
- Efficient resizing: image pyramids
- Aliasing - problem when structures in an image change when we resize it by removing every other pixel
- Nyquist theorem - if we want to reconstruct all frequencies up to $f$, we have to sample with the frequency of at least $2f$
- Gaussian pyramid
  - Smooth by a small filter
  - Resample

# Derivatives and edge detection
## Image derivatives
- Derivative convolution kernel 
  - Horizontal $\begin{bmatrix}
      -1 & 1
  \end{bmatrix}$
  - Vertical $\begin{bmatrix}
    -1 \\
    1
  \end{bmatrix}$
- Gradient points in the direction of the greatest intesnity change
  - Magnitude $\theta = \tan^{-1}(\dfrac{\frac{\partial f}{\partial y}}{\frac{\partial f}{\partial x}})$
  - Orientation $||\nabla f|| = \sqrt{\frac{\partial f}{\partial x}^2 + \frac{\partial f}{\partial y}^2}$
- With derivation, noise gets amplified $\implies$ smooth first
- Apply derivative kernel to Gaussian kernel

## From derivatives to edge detection
- "Optimal" edge detector
  - Good detection - minimum probability of false positives
  - Good localization - detected edges should be close to the location of true edges
  - Specificity - returns only a signle point per true edge
- Canny edge detector
  - Filter an image by derivative of a Gaussian
  - Calculate gradient magnitude and orientation
  - Thin potential edges to single pixel thickness
  - Select sequences of pixels that are likely an edge
- Thinning by non-maxima surpression
  - For each pixel, check if it's the local maximum in its gradient direction
  - Only local maxima should remain
- Hysteresis thresholding
  - Start tracing a line only at pixels that exceed the threshold $\text{k}_{\text{high}}$
  - Continue tracing if the pixels exceed the threshold $\text{k}_{\text{low}}$

## Edge detection by parametric models
- Hough transform
  - For each edge point compute parameters of all possible lines passing through it
  - For each set of parameters cast a vote
- Hough space: $(x, y) \rightarrow (m, b)$
  - $y = m_0y + b_0$
- Polar representation: $x\cos\theta - y\sin\theta = d$
  - $d$... distance from $[0,0]$
  - $\theta$...angle between $x$ axis and a perpendicular line from $[0,0]$ to our line
  - Sinusoid in Hough space
  - Voting: $H[d, \theta] += 1$
- Hough transform extensions
  - Use gradient direction instead of votin for all $\theta \in [0, 180]$
- Hough transform for circles
  - $(x_i - a)^2 + (y_i - b)^2 = r^2$
  - Cone in hough space; line if we know the gradient direction
  - Voting: $H[a, b, r] += 1$
- Generalized Hough transform
  - Define the shape model by edge points and a reference point
  - For each edge point calculate the displacement vector to the reference point 
  - Collect displacements in table, indexed by gradient direction
- Pros
  - Can detect shapes even if they are partially ocluded
  - Robust to noise
  - Can detect multiple shapes in one image
  - Can be easily modified to detect other shapes
- Cons
  - Can be computationally expensive (number of free parameters)
  - Requires a large amount of memory
  - May not be able to detect small shapes
  - May be sensitive to choice of parameters, for example the number of bins

# Fitting
- Least squares
  - Define the set of corresponding points {$x_i$}
  - Define the linear transformation $f$
  - Define the per-point error and stack errors in the vector $\epsilon$
  - Rewrite the error in the form $\epsilon = Ap - b$
  - Solve by pseudoinverse $A^{\dagger}b$
- Weighted least squares
  - Weight each point with $w_i$
  - Calculate the weighted cost $E(p)$
  - Find the best parameters $\tilde{p}$
- Robust least squares
  - <span style="color:lime">Not covered</span>
- Iterative reweighted least squares
  - <span style="color:lime">Not covered</span>
- Constrained least squares
  - Seeking a paremeter that satisfies constraints
  - Minimize perpendicular distance
- RANSAC
  - Algorithm
    - Randomly select the smallest group of correspondences from which we can estimate the parameters of our model
    - Fit the parametric model $\tilde{p}$ to the selected correspondences
    - Project all other points and count the number of inliers
    - Remember the parameters $\tilde{p}_{\text{opt}}$ that maximize the number of inliers
  - Parameters
    - Required number of correspondences
      - Smallest number that allows estimating the model parameters
    - Threshold distance for identifying inliers
      - Choose such a threshold that the probability that an inlier falls bellow the threshold is `p_w = 0.95`
    - Number of iterations $N = \dfrac{\log(1-p)}{\log(1 - (1 - e)^s)}$
      - $p$ ... probability of drawing a sample with all inliers
      - $e$ ... proportion of outliers (probability of selecting an outlier at random)
      - $s$ ... sample size <span style="color:orange">**(?)**</span> 
  

# Keypoints and matching
## Single scale keypoint detection
- Harris corner detector
  - Gradient covariance matrix: $M = \begin{bmatrix}
      G(\sigma) * I_x^2 & G(\sigma) * I_x I_y \\
      G(\sigma) * I_x I_y & G(\sigma) * I_y^2 
  \end{bmatrix}$  
  - Corners are detected by analysing $M$
    - Decompose into eigenvectors and eigenvalues: $M = R \begin{bmatrix}
        \lambda_\text{max} & 0 \\
        0 & \lambda_\text{min}
    \end{bmatrix} R^T$
      - $R$ ... eigenvectors
      - $\lambda$ ... eigenvalues
    - A corner has a strong gradient in the direction of both eigenvectors
    - Problem: eigenvalue calculation is expensive $\implies$ we use the ratio of both eigenvalues and an estimate of their magnitude
      - $\det(M) - \alpha \text{trace}^2(M) = 0$
      - Left side is the corner response function
- Hessian corner detector
  - $\text{Hessian}(I) = \begin{bmatrix}
      I_{xx} & I_{xy} \\
      I_{xy} & I_{yy}
  \end{bmatrix}$
  - $\det(\text{Hessian}(I)) = I_{xx}I_{yy} - I_{xy}^2$
  - Also finds blobs <span style="color:orange">**(?)**</span> 
- The corner resposnse function is:
  - invariant to rotation
  - not invariant to scale

## Scale selection
- Construct a scale function that outputs the same value for regions with the same content, even at different scales
- Automatic scale selection - select the scale that maximizes the scale function
- Laplacian of a Gaussian: $\nabla^2g = \dfrac{\partial^2 g}{\partial x^2} + \dfrac{\partial^2 g}{\partial y^2}$
  - Blob detector
  - Characteristic scale - the scale at which the LoG filter yields a maximum response
  - LoG can be approximated with difference of Gaussians: $DoG = G(x, y, k\sigma) - G(x, y, \sigma)$
    - Does not require partial derivatives
    - Result of Gaussian filtering already calculated during calculation of image resizing
    - DoG pyramid 
      - $L_i = G_i - upscale(G_{i+1})$
    - Keypoint localization using DoG
      - Find local maxima of DoG in scale space
      - Remove the low contrast points
      - Remove points detected at the edges

## Local desciptors
- Simplest descriptor: vector of region intensities
- SIFT
  - Split region into 16 sub-regions
  - Calculate gradient on each pixel and smooth over a few neighbours
  - In each cell calculate a histogram of gradient orientations
    - Each cell contributes with a weight proportional to its gradient magnitude
    - The contribution is weighted by a Gaussian centered at the region center
  - Stack histograms into a vector and normalize $\implies$ descriptor
- SIFT orientation normalization
  - Calculate the histograms of orientations
    - 36 bins by angle, each point contributes proportionally to its gradient magnitude and distance from the center
  - Determine the dominant orientation from histogram
  - Normalize - rotate gradients into a rectified orientation
  - Calculate the SIFT using these rectified gradients
- Affien adaptation
  - Problem: determine the characteristic shape of a local region
  - Solution:
    - In circular window calculate a gradient covariance matrix (similar to Harris)
    - Estimate an ellipse from the covariance matrix
    - Using the new window calculate the new covariance matrix and iterate
  - Affine patch normalization: transform, rotate, scale to transform the elipse into a circle
- Correspondences using keypoints
  - Strategies
    - For each keypoint in the left image, find the most similar keypoint in the right image
    - Keep only symmetric matches
    - Calculate the similarity of A to the second-most similar keypoint and the most similar keypoint and in the right image
      - Ratio of these two similarities will be low for distinctive key-points and high for non-distinctive ones
      - Threshold ~0.8 is good for SIFT
  - Algorithm
    - Detect keypoints
    - Determine potential correspondences
    - Reject improbable correspondences by strategy 1, 2 or 3
    - Perform RANSAC to fin inliers

# Camera geometry
- Extrinsic projection - 3D world to 3D camera
- Intrinsic projection - 3D camera to 2D image
- Homogeneneous coordinates: $\begin{bmatrix}
    x \\ y
\end{bmatrix} \rightarrow \begin{bmatrix}
    x \\ y \\ 1
\end{bmatrix}$
  - Point at infinity: $\begin{bmatrix}
    x \\ y \\ 0
  \end{bmatrix}$
- Principal axis - a line from camera center perpendicular to the image plane
- Principal point - a point where the principal axis punctures the image plane
- Normalized (camera) coordinate system - 2D system with origin at the principal point
- Projection matrix $P_0 = \begin{bmatrix}
  \alpha_x & 0 & x_0 & 0\\
  0 & \alpha_y & y_0 & 0 \\
  0 & 0 & 1 & 0
\end{bmatrix}$
  - Meaning of elements 
    - $f$ ... focal length
    - $m_x = \dfrac{M_x}{W_s}$
      - $M_x$ ... pixel matrix width
      - $W_s$ ... senzor width
      - Similar for $m_y$ 
    - $\alpha_x = m_x \cdot f$
- Calibration matrix $K = \begin{bmatrix}
    \alpha_x & s & x_0 \\
    0 & \alpha_y & y_0 \\
    0 & 0 & 1
\end{bmatrix}$
- The 3D camera coordinate system is related to the 3D world c.s. by a rotation matrix $R$ and translation $\tilde{t} = \tilde{C}$
  - $R$ ... how to rotate world c.s. to align it with camera c.s.
  - $\tilde{C}$ ... camera origin in world c.s.
  - $\tilde{X}$ ... point in 3D world c.s.
  - $\tilde{X}_{cam}$ ... $\tilde{X}$, but written in 3D camera c.s.
- World to camera c.s. transform: $\tilde{X}_{cam} = R(\tilde{X} - \tilde{C})$
- World to pixels transform: $x = K [I | 0]X_{cam} = K [R | - R\tilde{C}]X = PX$
  - $P = K[R|t]$
  - $t = -R\tilde{C}$
- $[a_\times] \equiv \begin{bmatrix}
  0 & -a_z & a_y \\
  a_z & 0 & -a_x \\
  -a_y & a_x & 0
\end{bmatrix}$
- Homography - plane-to-plane projection
  - Can be estimated by direct linear transform (DLT)
    - $Hx_r = x_t'$
    - $H$ ... matrix of $h_{i,j}$
    - $h$ ... $H$ in vector form
    - $Ah = 0$
    - $A = \begin{bmatrix}
      x_{r1} & y_{r1} & 1 & 0 & 0 & 0 & -x_{t1}x_{r1} & -x_{t1}y_{r1} & -x_{t1} \\
      0 & 0 & 0 & x_{r1} & y_{r1} & 1 & -y_{t1}x_{r1} & -y_{t1}y_{r1} & -y_{t1} \\
      \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
    \end{bmatrix}$
    - $A \stackrel{svd}{=} UDV^T$
    - $h = \dfrac{[v_{1,9}, \dots, v_{9,9}]^T}{v_{9,9}}$
    - Reshape $h$ to $H$
  - DLT works well if correspondences are normalized seperately in each view
    - Transformation $T_{pre}$ 
      - Subtract the average
      - Scale to average distance 1
      - $T_{pre} = \begin{bmatrix}
          a & 0 & c \\
          0 & b & d \\
          0 & 0 & 1
      \end{bmatrix}$
      - $\tilde{x} = T_{pre}x$
  - Homography estimation
    - Apply preconditioning to points in each image seperately
    - DLT
    - Transform back to remove preconditioning: $H = T_{pre}^{'-1}\tilde{H}T_{pre}$
- Vanishing points
  - $v = \begin{bmatrix}
    fX_D / Z_D \\
    fY_D / Z_D
  \end{bmatrix}$
  - Meaning of params
    - $f$ ... focal length
    - $D = \begin{bmatrix}
      X_D \\ Y_D \\ Z_D
    \end{bmatrix}$ ... vector on a parallel line
  - The horizon is formed by connecting the vanishing points of a plane
- Camera calibration
  - Estimate the projection matrix from a know callibration object
  - Figure out the internal and external parameters of the projection matrix: $P = P_\text{int}P_\text{ext}=K[R|t]$
  

# Multiview geometry
## Stereo geometry and scene reconstruction
- Find the intersection of two visual rays, corresponding to points $x_1$ and $x_2$
  - $x_1 \times P_1X = 0$
  - $x_2 \times P_2X = 0$
  - $AX = 0$
    - Solve by SVD
    - Solution for $X$ is the eigenvector corresponding to the smallest eigenvalue
- Nonlinear approach
  - Find $X$ that minimizes the reprojection errors
  - $d^2(x_1, P_1X) + d^2(x_2, P_2X)$
  - Iterative algorithm
    - Initialize by DLT
    - Optimize by gradient descent, Gauss-Newton, ...
- In general correspondences are unknown
- Potential matches for $p$ lie on the corresponding epipolar line $l'$
- Callibrated system
  - Essential matrix $E$ relates the corresponding image points; $E = [T_{\times}R]$
  - A 3D point is mapped to points $p$ and $p'$ which are related by $p^TEp' = 0$
    - $l' = (p^TE)^T$
    - $l = Ep'$
- Noncalibrated system
  - Fundamental matrix $F = K^{-T}EK'^{-1}$
- Baseline - a line connecting the camera centers
- Epipole - a point where the baseline punctures the image plane
- Epipolar plane - a plane connecting two epipoles and a 3D point
- Epipolar line - intersection of epipolar plane and image plane
- All epipolar lines of a single image intersect at the camera epipole
- Disparity - relates the right image to the left
  - Disparity estimation
    - Select a patch centered at a pixel in the left image
    - Compare to all patches in the right image along the epipolar line (in some window)
    - Select the patch with greatest similarity
    - Disparity = difference in position of patches
  - Semi global block matching
    - Apply line-based optimization across several directions in the image
    - Aggregate disparity energies from all direction-optimal assignments and take the disparity at each pixel that recieved a minimum energy

## Structure from motion
- Given several images of a scene, reconstruct all camera positions and the 3D scene
- SFM pipeline
  1. Keypoints
  2. Matches
  3. Fundamental matrix
  4. Essential matrix
  5. $[R|t]$
  6. Triangulation
- Fundamental matrix estimation
  - Find keypoints
  - Find correspondences using proximity constraints
  - Filter the correspondences by visual similarity
  - RANSAC

## Active stereo
- Project "structured" light patterns over an object

# Recognition 1
## Principal component analysis
- Find a low-dimensional subspace that efficiently compresses the data
- Reconstruction error minimization is equivalent to maximization of variance of projected points
- Variance $\text{var}(a) = u^T \Sigma u \rightarrow$ find $u$ that maximizes $\text{var}(a)$
  - Write a Lagrangian for constrained optimization $\implies$ $\Sigma u = \lambda u$ 
  - $\text{var}(a)$ is maximized by the largest eigenvalue
- PCA is actually a change of coordinate system that captures major directions of the variance in the data
- Projection of data $x_i$ into the new c.s.: $y_i = U^T(x_i - \mu)$
- Projection of $y_i$ back into the original c.s.: $x_i = U y_i + \mu$
  - $U = [u_1, u_2]$ ... matrix of eigenvectors
- PCA algorithm (data $X$ is a matrix of size $M \times N$, $M$ ... training sample size, $N$ ... number of training samples)
  - Estimate the mean vector $\mu = \frac{1}{N}\sum_{i = 1}^{N}x_i$
  - Center the input data around the mean $\hat{X}_i = X_i - \mu$
  - If $M \le N$
    - Estimate the covariance matrix $C = \frac{1}{N}\hat{X}\hat{X}^T$
    - Perform SVD on $C$ and get eigenvectors $U$ and eigenvalues $\lambda$
  - else
    - Estimate the inner product matrix: $C' = \frac{1}{N}\hat{X}^T\hat{X}$
    - Perform SVD on $C'$ and get $U'$ and $\lambda'$
    - Eigenvectors $U: u_i = \dfrac{\hat{X}u_i'}{\sqrt{N\lambda_i'}}, i = 1...N$
    - Eigenvalues $\lambda = \lambda'$
- PCA is a linear autoencoder

## Linear discriminant analysis
- Assume we know the class labels
- Task: derive an approach that takes the class labels into consideration in subspace estimation
- Find a subspace that:
  - Maximizes the distance between classes
  - Minimizes the distance within classes

# Recognition 2 and object detection
## Handcrafted nonlinear transforms
- Gradient-based representation - encode a local gradient distribution using histograms
- SVM

## Learning features by feature selection
- Boosting
- AdaBoost
- Cascade of classifiers - apply first few classifiers to reject the windows that obviously don't contain the particualar category. Then re-classify the regions that survived with strong classifiers.

## End-to-end feature learning
- CNN

# Recognition and detection using local features
## Bag of words model
- Summarize an image by distribution over visual words
- Feature detection 
  - Feature points
  - Normalize each region to remove local geometric deformations
- A SIFT descriptor is a word in high dimensional space
- Clustering
  - K-means
  - Center of each cluster is the visual word
  - Learn the code book on separate training data
  - Apply code book for feature quantization
    - Takes a feature vector and maps it to the index of the closest code vector
- Image representation
  - Normalized histogram (?)
- Build a classfier
  - SVM
- Recognition
  - Encode an image with the learned dictionary
- R-CNN
  - Get region with selective search
  - Crop region and send it to a CNN
  - Result: classification and box regression
  - Problem: run each region through CNN $\implies$ slow
-  Fast R-CNN
   -  Extract features from whole image (use a feature extraction network)
   -  For a region, cut out a volume from feature space and classify (use a multilayer perceptron - MLP)
   -  Problem: selective search is slow
- Faster R-CNN
  - Replace selective search by a region  proposal network (RPN)
  - Feature pyramid: extract features at multiple scales and merge them
- Mask R-CNN
  - After extracting the bounding box, predict the segmentation mask of the object
  
## Object detection by feature constellations
- Represent target model in terms of small "parts" that can be detected even under an affine deformatin
- Detection by Generalized Hough Transform
  - Index descriptors
  - Apply GHT to obtaiin approximate detections
  - Refine each detection by fitting affine transform between the points on the object and the detected points from GHT 


---

# Q&A

**When is a 2D filter seperable?**
> A 2D filter is separable when it can be expressed as the outer product of two 1D filters. This means that we first convolve the image with one 1D filter along the rows and then convolve the result along the columns.

**Is Harris detector scale / rotation invariant. What about DoG? Why/Why not?**
> The Harris corner detector is not scale-invariant, meaning that it may not be able to detect corners at different scales or sizes in an image. However, it is rotation-invariant, meaning that it can detect corners regardless of the angle at which they are oriented in the image. The Difference of Gaussian (DoG) detector, on the other hand, is both scale- and rotation-invariant. This is because the DoG detector works by taking the difference of two Gaussian-smoothed versions of the image, which effectively creates a scale-space representation of the image. This allows the detector to find features at different scales and orientations in the image.

**What's the difference between the projection, calibration, homography, fundamental and essential matrices?**

> Projection matrix - for projecting a 3D point in camera coordinate system to a 2D image plane point (pixels)
> $P = \begin{bmatrix}
    \alpha_x & s & x_0 & 0 \\
    0 & \alpha_y & y_0 & 0 \\
    0 & 0 & 1 & 0
\end{bmatrix}$ 
> $\alpha_x, \alpha_y$ ... focal length in pixels;
> $x_0, y_0$ ... principal point coordinates in pixels;
> $s$ ... skewing parameter (0 for rectangular grid)

> Calibration matrix - Contains all intrinsic parameters of a camera and tells us how a 3D point in camera coordinate system is projected into pixels. Similar to projection matrix.
> $K = \begin{bmatrix}
    \alpha_x & s & x_0 \\
    0 & \alpha_y & y_0\\
    0 & 0 & 1
\end{bmatrix}$ 
>
> Homography matrix - matrix that shows the relationship of the same planar object in two images
>
> Essential matrix - relates the translation and rotation of two cameras. It is used to obtain epipolar lines.
> $E = [T_\times]R$; $T$... translation matrix of one camera relative to the other; $R$ ... rotation matrix of one camera relative to the other
>
> Fundamental matrix - relates corresponding points between two camera. It is used to establish correspondences between two images and to recover a 3D scene.

**Describe the principle of how a sharpening filter with smoothing works.**
> Convole with a Gaussian filter to smoothen. For sharpening we compare the smoothed image with the original. If the difference is big enough, we subtract them.

**What is a weak calibration of a stereo system? Briefly describe its steps.**
Weak calibration of a stereo system involves finding intrinsic and extrinsic parameters of two cameras.
1. Intrinsic parameters
2. Find relative position and orientation of cameras
3. Find correspondences
4. Triangulation

# Formulas

## Perspective projection equation

$\dfrac{f}{-y'} = \dfrac{-z}{y}$
- $f$ ... focal length
- $y$ ... object height
- $y'$ ... object height on film
- $z$ ... distance from object to camera

## RANSAC - number of iterations
$N = \dfrac{\log(1-p)}{\log(1 - (1 - e)^s)}$
- $p$ ... probability of drawing a sample with all inliers
- $e$ ... proportion of outliers (probability of selecting an outlier at random)
- $s$ ... sample size

## Disparity
$d = x_1 -x_2 = \dfrac{fT}{p_z}$
- $d$ ... disparity
- $x_1, x_2$ ... position of the point in left and right camera 
- $f$ ... focal length
- $T$ ... baseline - distance between cameras
- $p_z$ ... $z$ coordinate of the point

## Fundamental matrix projection
$x_L \cdot F \cdot x_R$