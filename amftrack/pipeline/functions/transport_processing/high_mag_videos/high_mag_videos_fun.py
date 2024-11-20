import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from scipy.ndimage import convolve

from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_sparse_to_graph,
    generate_nx_graph,
)
import scipy
from amftrack.pipeline.functions.image_processing.node_id import remove_spurs
from amftrack.pipeline.functions.image_processing.extract_skel import (
    remove_component,
    remove_holes,
)
import numpy as np
from amftrack.pipeline.functions.image_processing.extract_width_fun import (
    generate_pivot_indexes,
    compute_section_coordinates,
)
from skimage.measure import profile_line
from amftrack.pipeline.functions.image_processing.experiment_class_surf import orient
from skimage.filters import frangi, threshold_yen
from scipy.optimize import minimize_scalar
from skimage.morphology import skeletonize
import itertools, operator
import gc


def get_length_um_edge(edge, nx_graph, space_pixel_size):
    pixel_conversion_factor = space_pixel_size
    length_edge = 0
    pixels = nx_graph.get_edge_data(*edge)["pixel_list"]
    for i in range(len(pixels) // 10 + 1):
        if i * 10 <= len(pixels) - 1:
            length_edge += np.linalg.norm(
                np.array(pixels[i * 10])
                - np.array(pixels[min((i + 1) * 10, len(pixels) - 1)])
            )
    #             length_edge+=np.linalg.norm(np.array(pixels[len(pixels)//10-1*10-1])-np.array(pixels[-1]))
    return length_edge * pixel_conversion_factor


def calcGST(inputIMG, w):
    """
    Calculates the Image orientation and the image coherency. Image orientation is merely a guess, and image coherency gives an idea how sure that guess is.
    inputIMG:   The input image
    w:          The window size of the various filters to use. Large boxes catch higher order structures.
    """

    # The idea here is to perceive any patch of the image as a transformation matrix.
    # Such a matrix will have some eigenvalues, which describe the direction of uniform transformation.
    # If the largest eigenvalue is much bigger than the smallest eigenvalue, that indicates a strong orientation.

    img = inputIMG.astype(np.float32)
    imgDiffX = cv2.Sobel(img, cv2.CV_32F, 1, 0, -1)
    imgDiffY = cv2.Sobel(img, cv2.CV_32F, 0, 1, -1)
    imgDiffXY = cv2.multiply(imgDiffX, imgDiffY)
    imgDiffXX = cv2.multiply(imgDiffX, imgDiffX)
    imgDiffYY = cv2.multiply(imgDiffY, imgDiffY)
    J11 = cv2.boxFilter(imgDiffXX, cv2.CV_32F, (w, w))
    J22 = cv2.boxFilter(imgDiffYY, cv2.CV_32F, (w, w))
    J12 = cv2.boxFilter(imgDiffXY, cv2.CV_32F, (w, w))
    tmp1 = J11 + J22
    tmp2 = J11 - J22
    tmp2 = cv2.multiply(tmp2, tmp2)
    tmp3 = cv2.multiply(J12, J12)
    tmp4 = np.sqrt(tmp2 + 4.0 * tmp3)
    lambda1 = 0.5 * (tmp1 + tmp4)  # biggest eigenvalue
    lambda2 = 0.5 * (tmp1 - tmp4)  # smallest eigenvalue
    imgCoherencyOut = cv2.divide(lambda1 - lambda2, lambda1 + lambda2)
    imgOrientationOut = cv2.phase(J22 - J11, 2.0 * J12, angleInDegrees=True)
    imgOrientationOut = 0.5 * imgOrientationOut
    return imgCoherencyOut, imgOrientationOut


def segment(images_adress, threshold=10):
    images = [imageio.imread(file) for file in images_adress]
    images = [cv2.resize(image, np.flip(images[0].shape)) for image in images]
    average_proj = np.mean(np.array(images), axis=0)
    segmented = average_proj > threshold
    segmented = remove_holes(segmented)
    segmented = segmented.astype(np.uint8)
    connected = remove_component(segmented)
    connected = connected.astype(np.uint8)
    skeletonized = cv2.ximgproc.thinning(np.array(connected, dtype=np.uint8))
    skeleton = scipy.sparse.dok_matrix(skeletonized)
    nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
    nx_graph, pos = remove_spurs(nx_graph, pos, threshold=20)
    # nx_graph = clean_degree_4(nx_graph, pos)[0]
    return (skeletonized, nx_graph, pos)


def extract_section_profiles_for_edge(
    edge: tuple,
    pos: dict,
    raw_im: np.array,
    nx_graph,
    resolution=5,
    offset=4,
    step=15,
    target_length=120,
    bounds=(0, 1),
) -> np.array:
    """
    Main function to extract section profiles of an edge.
    Given an Edge of Experiment at timestep t, returns a np array
    of dimension (target_length, m) where m is the number of section
    taken on the hypha.
    :param resolution:  distance between two measure points along the hypha
    :param offset:      distance at the end and the start where no point is taken
    :param step:        step in pixel to compute the tangent to the hypha
    :target_length:     length of the section extracted in pixels
    :return:            np.array of sections, list of segments in TIMESTEP referential
    """
    pixel_list = orient(nx_graph.get_edge_data(*edge)["pixel_list"], pos[edge[0]])
    offset = max(
        offset, step
    )  # avoiding index out of range at start and end of pixel_list
    pixel_indexes = generate_pivot_indexes(
        len(pixel_list), resolution=resolution, offset=offset
    )

    # WARNING: Below function is the bottleneck if extract_section_profiles_for_edge is called multiple times for the
    # same edge. Consider implementing this code on a higher level.
    list_of_segments = compute_section_coordinates(
        pixel_list, pixel_indexes, step=step, target_length=target_length + 1
    )  # target_length + 1 to be sure to have length all superior to target_length when cropping
    # TODO (FK): is a +1 enough?
    images = {}
    l = []
    im = raw_im

    for i, sect in enumerate(list_of_segments):
        point1 = np.array([sect[0][0], sect[0][1]])
        point2 = np.array([sect[1][0], sect[1][1]])
        profile = profile_line(im, point1, point2, mode="constant")[
            int(bounds[0] * target_length) : int(bounds[1] * target_length)
        ]
        profile = profile.reshape((1, len(profile)))
        # TODO(FK): Add thickness of the profile here
        l.append(profile)
    return np.concatenate(l, axis=0), list_of_segments


def plot_segments_on_image(segments, ax, color="red", bounds=(0, 1), alpha=1, adj=1):
    for point1_pivot, point2_pivot in segments:
        point1 = (1 - bounds[0]) * point1_pivot + bounds[0] * point2_pivot
        point2 = (1 - bounds[1]) * point1_pivot + bounds[1] * point2_pivot
        ax.plot(
            [point1[1] * adj, point2[1] * adj],  # x1, x2
            [point1[0] * adj, point2[0] * adj],  # y1, y2
            color=color,
            linewidth=2,
            alpha=alpha,
        )


# OLD FUNCTION. Use get_kymo_new
# def get_kymo(
#         edge,
#         pos,
#         images_adress,
#         nx_graph_pruned,
#         resolution=1,
#         offset=4,
#         step=15,
#         target_length=10,
#         bound1=0,
#         bound2=1,
# ):
#     kymo = []
#     for image_adress in images_adress:
#         image = imageio.imread(image_adress)
#         slices, segments = extract_section_profiles_for_edge(
#             edge,
#             pos,
#             image,
#             nx_graph_pruned,
#             resolution=resolution,
#             offset=offset,
#             step=step,
#             target_length=target_length,
#             bound1=bound1,
#             bound2=bound2,
#         )
#         kymo_line = np.mean(slices, axis=1)
#         kymo.append(kymo_line)
#     return np.array(kymo)


def filter_kymo_left_old(kymo):
    A = kymo[:, :]
    B = np.flip(A, axis=0)
    C = np.flip(A, axis=1)
    D = np.flip(B, axis=1)
    tiles = [[D, B, D], [C, A, C], [D, B, D]]
    tiles = [cv2.hconcat(imgs) for imgs in tiles]
    tiling_for_fourrier = cv2.vconcat(tiles)
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(tiling_for_fourrier))
    coordinates_middle = np.array(dark_image_grey_fourier.shape) // 2
    LT_quadrant = np.s_[: coordinates_middle[0], : coordinates_middle[1]]
    LB_quadrant = np.s_[coordinates_middle[0] + 1 :, : coordinates_middle[1]]
    RB_quadrant = np.s_[coordinates_middle[0] + 1 :, coordinates_middle[1] :]
    RT_quadrant = np.s_[: coordinates_middle[0], coordinates_middle[1] :]

    filtered_fourrier = dark_image_grey_fourier
    filtered_fourrier[LT_quadrant] = 0
    filtered_fourrier[RB_quadrant] = 0
    filtered = np.fft.ifft2(filtered_fourrier)
    shape_v, shape_h = filtered.shape
    shape_v, shape_h = shape_v // 3, shape_h // 3
    middle_slice = np.s_[shape_v : 2 * shape_v, shape_h : 2 * shape_h]
    middle = filtered[middle_slice]
    filtered_left = A - np.abs(middle)
    return filtered_left


def tile_image(img):
    A = img[:, :]
    B = np.flip(A, axis=0)
    C = np.flip(A, axis=1)
    D = np.flip(B, axis=1)
    tiles = [[D, B, D], [C, A, C], [D, B, D]]
    tiles = [cv2.hconcat(imgs) for imgs in tiles]
    tiling_for_fourrier = cv2.vconcat(tiles)
    return tiling_for_fourrier


def create_paved_kymograph(kymo):
    height, width = kymo.shape

    # Duplicate and manipulate images
    subFourier1 = kymo.copy()
    subFourier2 = cv2.flip(kymo, 1)  # Flip horizontally
    subFourier3 = cv2.flip(subFourier2, 0)  # Flip vertically after horizontal
    subFourier4 = cv2.flip(kymo, 0)  # Flip vertically

    # Create a larger image and place manipulated images accordingly
    filter_forward = np.zeros((3 * height, 3 * width), dtype=kymo.dtype)
    filter_forward[height : 2 * height, width : 2 * width] = subFourier1
    filter_forward[height : 2 * height, 0:width] = subFourier2
    filter_forward[height : 2 * height, 2 * width : 3 * width] = subFourier2
    filter_forward[0:height, 0:width] = subFourier3
    filter_forward[0:height, 2 * width : 3 * width] = subFourier3
    filter_forward[2 * height : 3 * height, 0:width] = subFourier3
    filter_forward[2 * height : 3 * height, 2 * width : 3 * width] = subFourier3
    filter_forward[0:height, width : 2 * width] = subFourier4
    filter_forward[2 * height : 3 * height, width : 2 * width] = subFourier4
    return filter_forward


def apply_fourier_operations(tiled_image):
    dft = np.fft.fftshift(
        np.fft.fft2(tiled_image)
    )  # Apply FFT and shift zero frequency to center
    # Zero out specific regions in the Fourier transform
    h, w = dft.shape
    # Horizontal line across the middle
    dft[h // 2 - 1 : h // 2 + 1, :] = 0
    # Top-left quadrant
    dft[: h // 2, : w // 2] = 0
    # botom-right quadrant
    dft[h // 2 :, w // 2 :] = 0

    # Inverse Fourier Transform
    inverse_dft = np.fft.ifft2(np.fft.ifftshift(dft)).real

    return inverse_dft


def filter_kymo_left(kymo):
    height, width = kymo.shape
    paved_kymo = create_paved_kymograph(kymo)
    paved_kymo_filter = apply_fourier_operations(paved_kymo)
    paved_kymo = create_paved_kymograph(kymo)
    filtered_kymo = paved_kymo_filter[height : 2 * height, width : 2 * width]
    filtered_kymo -= np.percentile(filtered_kymo, 10)
    # filtered_kymo[np.where(filtered_kymo<0)] =0

    return filtered_kymo

    """
    Simple function that filters the kymograph and outputs the forward and backward filters.
    """


def filter_kymo(kymo):
    filtered_left = filter_kymo_left(kymo)
    filtered_right = np.flip(filter_kymo_left(np.flip(kymo, axis=1)), axis=1)
    return (filtered_left, filtered_right)


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def get_speeds(kymo, W, C_Thr, fps, binning, magnification):
    time_pixel_size = 1 / fps  # s.pixel

    space_pixel_size = 2 * 1.725 / (magnification) * binning  # um.pixel
    imgCoherency, imgOrientation = calcGST(kymo, W)
    nans = np.empty(imgOrientation.shape)
    nans.fill(np.nan)

    real_movement = np.where(imgCoherency > C_Thr, imgOrientation, nans)
    speed = (
        np.tan((real_movement - 90) / 180 * np.pi) * space_pixel_size / time_pixel_size
    )  # um.s-1


def get_width_from_graph_im(edge, pos, image, nx_graph_pruned, slice_length=400):
    bound1 = 0
    bound2 = 1
    offset = 100
    step = 30
    target_length = slice_length
    resolution = 1
    slices, _ = extract_section_profiles_for_edge(
        edge,
        pos,
        image,
        nx_graph_pruned,
        resolution=resolution,
        offset=offset,
        step=step,
        target_length=target_length,
        bounds=(bound1, bound2),
    )
    return get_width(slices)


def get_width(slices, avearing_window=50, num_std=2):
    widths = []
    for index in range(len(slices)):
        thresh = np.mean(
            (
                np.mean(slices[index, :avearing_window]),
                np.mean(slices[index, -avearing_window:]),
            )
        )
        std = np.std(
            (
                np.concatenate(
                    (slices[index, :avearing_window], slices[index, -avearing_window:])
                )
            )
        )
        try:
            deb = np.min(np.argwhere(slices[index, :] < thresh - num_std * std))
            end = np.max(np.argwhere(slices[index, :] < thresh - num_std * std))
            print(deb)
            width = end - deb
            widths.append(width)
        except ValueError:
            continue
    return np.median(widths)


def find_histogram_edge(image, plot=False):
    # Calculate the histogram
    hist, bins = np.histogram(image.flatten(), 40)
    hist = hist.astype(float) / hist.max()  # Normalize the histogram

    # Sobel Kernel
    sobel_kernel = np.array([-1, 0, 1])

    # Apply Sobel edge detection to the histogram
    sobel_hist = convolve(hist, sobel_kernel)

    # Find the point with the highest gradient change
    threshold = np.argmax(sobel_hist)

    # Optional: Plot the results
    if plot:
        plt.figure(figsize=(10, 5))

        # Plot the original histogram
        plt.subplot(1, 2, 1)
        plt.plot(hist)
        plt.axvline(x=threshold, color="r", linestyle="--")

        plt.title("Histogram")

        # Plot the Sobel histogram
        plt.subplot(1, 2, 2)
        plt.plot(sobel_hist)
        plt.title("Sobel Histogram")
        plt.axvline(x=threshold, color="r", linestyle="--")

        plt.show()

    return bins[threshold]


def calculate_renyi_entropy(threshold, pixels):
    # Calculate probabilities and entropies
    Ps = np.mean(pixels <= threshold)
    Hs = -np.sum(
        pixels[pixels <= threshold] * np.log(pixels[pixels <= threshold] + 1e-10)
    )
    Hn = -np.sum(pixels * np.log(pixels + 1e-10))

    # Calculate phi(s)
    phi_s = np.log(Ps * (1 - Ps)) + Hs / Ps + (Hn - Hs) / (1 - Ps)

    return -phi_s


def RenyiEntropy_thresholding(image):
    # Flatten the image
    pixels = image.flatten()

    # Find the optimal threshold
    initial_threshold = np.mean(pixels)
    result = minimize_scalar(
        calculate_renyi_entropy, bounds=(0, 255), args=(pixels,), method="bounded"
    )

    # The image is rescaled to [0,255] and thresholded
    optimal_threshold = result.x
    _, thresholded = cv2.threshold(
        image / np.max(image) * 255, optimal_threshold, 255, cv2.THRESH_BINARY
    )

    return thresholded


def segment_brightfield_std(images, seg_thresh=1.10, threshtype="hist_edge"):
    """
    Segmentation method for brightfield video, uses vesselness filters to get result.
    image:          Input image
    thresh:         Value close to zero such that the function will output a boolean array
    threshtype:     Type of threshold to apply to segmentation. Can be hist_edge, Renyi or Yen

    """
    std_image = np.std(images, axis=0) / np.mean(images, axis=0)
    smooth_im_blur = cv2.blur(std_image, (100, 100))
    if threshtype == "hist_edge":
        # the biggest derivative in the hist is calculated and we multiply with a small number to sit just right of that.
        thresh = find_histogram_edge(smooth_im_blur)
        segmented = (smooth_im_blur >= thresh * seg_thresh).astype(np.uint8) * 255

    elif threshtype == "Renyi":
        # this version minimizes a secific entropy (phi)
        segmented = RenyiEntropy_thresholding(smooth_im_blur)

    elif threshtype == "Yen":
        # This maximizes the distance between the two means and probabilities, sigma^2 = p(1-p)(mu1-mu2)^2
        thresh = threshold_yen(smooth_im_blur)
        segmented = (smooth_im_blur >= thresh).astype(np.uint8) * 255

    else:
        print("threshold type has a typo! rito pls fix.")
    skeletonized = skeletonize(segmented > 0)

    skeleton = scipy.sparse.dok_matrix(skeletonized)
    nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
    nx_graph_pruned, pos = remove_spurs(nx_graph, pos, threshold=200)

    return (segmented, nx_graph_pruned, pos)


def incremental_mean_std(images):
    n = len(images)
    sum_images = None
    sum_sq_diff = None

    for image in images:
        if sum_images is None:
            sum_images = np.zeros_like(image, dtype=np.float64)
            sum_sq_diff = np.zeros_like(image, dtype=np.float64)

        sum_images += image

    mean_image = sum_images / n

    for image in images:
        sq_diff = (image - mean_image) ** 2
        sum_sq_diff += sq_diff

    variance_image = sum_sq_diff / n
    std_dev_image = np.sqrt(variance_image)

    return mean_image, std_dev_image


def incremental_mean_std_address(image_addresses):
    n = len(image_addresses)
    sum_images = None
    sum_sq_diff = None

    for address in image_addresses:
        image = imageio.imread(address)
        if sum_images is None:
            sum_images = np.zeros_like(image, dtype=np.float32)
            sum_sq_diff = np.zeros_like(image, dtype=np.float32)

        sum_images += image
        # del image  # Suggest deletion of the image variable
        # gc.collect()
    mean_image = sum_images / n

    for address in image_addresses:
        image = imageio.imread(address)
        sq_diff = (image - mean_image) ** 2
        sum_sq_diff += sq_diff
        # del image  # Suggest deletion of the image variable
        # gc.collect()

    variance_image = sum_sq_diff / n
    std_dev_image = np.sqrt(variance_image)

    return mean_image, std_dev_image


def segment_brightfield_ultimate(
    image_address,
    seg_thresh=1.15,
):
    """
    Segmentation method for brightfield video.
    image:          Input image
    thresh:         Value close to zero such that the function will output a boolean array
    threshtype:     Type of threshold to apply to segmentation. Can be hist_edge, Renyi or Yen

    """
    mean_image, std_image = incremental_mean_std_address(image_address)
    smooth_im_blur = cv2.blur(std_image, (20, 20))
    smooth_im_blur_mean = cv2.blur(mean_image, (20, 20))

    CVs = smooth_im_blur / smooth_im_blur_mean
    CVs_blurr = cv2.blur(CVs, (20, 20))
    thresh = find_histogram_edge(CVs_blurr)

    segmented = (CVs_blurr >= thresh * seg_thresh).astype(np.uint8) * 255
    skeletonized = skeletonize(segmented > 0)

    skeleton = scipy.sparse.dok_matrix(skeletonized)
    nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
    nx_graph_pruned, pos = remove_spurs(nx_graph, pos, threshold=200)

    return (segmented, nx_graph_pruned, pos)


def segment_fluo_new(images, seg_thresh=1.10, threshtype="hist_edge"):
    """
    Segmentation method for brightfield video, uses vesselness filters to get result.
    image:          Input image
    thresh:         Value close to zero such that the function will output a boolean array
    threshtype:     Type of threshold to apply to segmentation. Can be hist_edge, Renyi or Yen

    """
    std_image = np.mean(images, axis=0)
    smooth_im_blur = cv2.blur(std_image, (20, 20))
    if threshtype == "hist_edge":
        # the biggest derivative in the hist is calculated and we multiply with a small number to sit just right of that.
        thresh = find_histogram_edge(smooth_im_blur)
        segmented = (smooth_im_blur >= thresh * seg_thresh).astype(np.uint8) * 255

    elif threshtype == "Renyi":
        # this version minimizes a secific entropy (phi)
        segmented = RenyiEntropy_thresholding(smooth_im_blur)

    elif threshtype == "Yen":
        # This maximizes the distance between the two means and probabilities, sigma^2 = p(1-p)(mu1-mu2)^2
        thresh = threshold_yen(smooth_im_blur)
        segmented = (smooth_im_blur >= thresh).astype(np.uint8) * 255

    else:
        print("threshold type has a typo! rito pls fix.")

    skeletonized = skeletonize(segmented > 0)

    skeleton = scipy.sparse.dok_matrix(skeletonized)
    nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
    nx_graph_pruned, pos = remove_spurs(nx_graph, pos, threshold=200)

    return (segmented, nx_graph_pruned, pos)


def segment_brightfield(
    image,
    thresh=0.5e-6,
    frangi_range=np.arange(70, 170, 30),
    seg_thresh=11,
    binning=2,
    close_size=None,
    thresh_adjust=0,
):
    """
    Segmentation method for brightfield video, uses vesselness filters to get result.
    image:          Input image
    thresh:         Value close to zero such that the function will output a boolean array
    frangi_range:   Range of values to use a frangi filter with. Frangi filter is very good for brightfield vessel segmentation

    """
    frangi_range = frangi_range * 2 / binning
    smooth_im_blur = cv2.blur(-image, (11, 11))
    smooth_im = frangi(-smooth_im_blur, frangi_range)
    smooth_im = np.array(smooth_im * (255 / np.max(smooth_im)), dtype=np.uint8)
    ret, segmented = cv2.threshold(
        smooth_im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(ret)
    if thresh_adjust != 0:
        ret, segmented = cv2.threshold(smooth_im, 0, 255, int(ret) + thresh_adjust)
    if close_size is not None:
        segmented = cv2.morphologyEx(
            segmented, cv2.MORPH_CLOSE, np.ones((close_size, close_size))
        )

    #     seg_shape = smooth_im.shape

    #     for i in range(1, 100):
    #         _, segmented = cv2.threshold(smooth_im, i, 255, cv2.THRESH_BINARY)
    #         coverage = 100 * np.sum(1 * segmented.flatten()) / (255 * seg_shape[0] * seg_shape[1])
    #         if coverage < seg_thresh:
    #             break

    skeletonized = skeletonize(segmented > seg_thresh)
    skeleton = scipy.sparse.dok_matrix(skeletonized)
    nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
    nx_graph_pruned, pos = remove_spurs(nx_graph, pos, threshold=200)
    return (segmented, nx_graph_pruned, pos)


def get_kymo_new(
    edge,
    pos,
    images_adress,
    nx_graph_pruned,
    resolution=1,
    offset=4,
    step=15,
    target_length=10,
    bounds=(0, 1),
    x_len=10,
    order=None,
):
    """
    The new get_kymo function. The old one had some inefficiencies that lead to it being much slower than this one.
    edge:               The edge in question. Will pull from skeleton the positional arguments
    pos:                Array of positions of nodes that are associated with the skeleton
    images_adress:      misspelled variable that gives the address of the image folder housing the video
    nx_graph_pruned:    Graph array that houses the skeleton of the hypha structure in the video
    resolution:         No idea what this does, probably important
    offset:             Measure of the length of the hypha
    step:               How large the step is between two points on the edge for tangent calculation.
                        Large step means smooth kymo, but less accuracy, small step means more jittery kymo, but closer to the hypha
    target_length:      Target width of the hypha. Adjust this to fit the edge of the hypha neatly into an edge video.
    bounds:             tuple describing what fraction of the total hypha length to use. Range is 0.0 (start) to 1.0 (end)
    order:              Important variable that is changed in the function as well. No idea what it does.
    """

    """
    Following section calculates the section coordinates for the skeleton of the hypha.
    These coordinates are used for all images in the video. Generally we don't expect hyphae to move.
    """
    pixel_list = orient(
        nx_graph_pruned.get_edge_data(*edge)["pixel_list"], pos[edge[0]]
    )
    offset = max(
        offset, step
    )  # avoiding index out of range at start and end of pixel_list
    pixel_indexes = generate_pivot_indexes(
        len(pixel_list), resolution=resolution, offset=offset
    )
    list_of_segments = compute_section_coordinates(
        pixel_list, pixel_indexes, step=step, target_length=target_length + 1
    )

    perp_lines = []
    kymo = []
    for i, sect in enumerate(list_of_segments):
        point1 = np.array([sect[0][0], sect[0][1]])
        point2 = np.array([sect[1][0], sect[1][1]])
        perp_lines.append(extract_perp_lines(point1, point2))

    if len(images_adress) < 500:
        for image_adress in images_adress:
            im = imageio.imread(image_adress)
            order = validate_interpolation_order(im.dtype, order)
            l = []
            for perp_line in perp_lines:
                pixels = ndi.map_coordinates(
                    im,
                    perp_line,
                    prefilter=order > 1,
                    order=order,
                    mode="reflect",
                    cval=0.0,
                )
                pixels = np.flip(pixels, axis=1)
                pixels = pixels[
                    int(bounds[0] * target_length) : int(bounds[1] * target_length)
                ]
                pixels = pixels.reshape((1, len(pixels)))
                l.append(pixels)

            slices = np.concatenate(l, axis=0)
            kymo_line = np.sum(slices, axis=1) / (target_length)
            kymo.append(kymo_line)
    else:
        for image_adress in images_adress[:499]:
            im = imageio.imread(image_adress)
            order = validate_interpolation_order(im.dtype, order)
            l = []
            for perp_line in perp_lines:
                pixels = ndi.map_coordinates(
                    im,
                    perp_line,
                    prefilter=order > 1,
                    order=order,
                    mode="reflect",
                    cval=0.0,
                )
                pixels = np.flip(pixels, axis=1)
                pixels = pixels[
                    int(bounds[0] * target_length) : int(bounds[1] * target_length)
                ]
                pixels = pixels.reshape((1, len(pixels)))
                l.append(pixels)

            slices = np.concatenate(l, axis=0)
            kymo_line = np.sum(slices, axis=1) / (target_length)
            kymo.append(kymo_line)
    return np.array(kymo)


def get_edge_image(
    edge,
    pos,
    images_address,
    nx_graph_pruned,
    resolution=1,
    offset=4,
    step=30,
    target_length=10,
    img_frame=0,
    bounds=(0, 1),
    order=None,
    logging=False,
):
    slices_list = []
    pixel_list = orient(
        nx_graph_pruned.get_edge_data(*edge)["pixel_list"], pos[edge[0]]
    )
    offset = max(
        offset, step
    )  # avoiding index out of range at start and end of pixel_list
    pixel_indexes = generate_pivot_indexes(
        len(pixel_list), resolution=resolution, offset=offset
    )
    list_of_segments = compute_section_coordinates(
        pixel_list, pixel_indexes, step=step, target_length=target_length + 1
    )
    perp_lines = []

    for i, sect in enumerate(list_of_segments):
        point1 = np.array([sect[0][0], sect[0][1]])
        point2 = np.array([sect[1][0], sect[1][1]])
        perp_lines.append(extract_perp_lines(point1, point2))

    if np.ndim(img_frame) == 0:
        im_list = [imageio.imread(images_address[img_frame])]
        order = validate_interpolation_order(im_list[0].dtype, order)
    else:
        im_list = [imageio.imread(images_address[frame]) for frame in img_frame]
        order = validate_interpolation_order(im_list[0].dtype, order)
    for im in im_list:
        l = []
        for perp_line in perp_lines:
            pixels = ndi.map_coordinates(
                im,
                perp_line,
                prefilter=order > 1,
                order=order,
                mode="reflect",
                cval=0.0,
            )
            pixels = np.flip(pixels, axis=1)
            pixels = pixels[
                int(bounds[0] * target_length) : int(bounds[1] * target_length)
            ]
            pixels = pixels.reshape((1, len(pixels)))
            # TODO(FK): Add thickness of the profile here
            l.append(pixels)
        slices = np.concatenate(l, axis=0)
        slices_list.append(slices)

    if np.ndim(img_frame) == 0:
        return slices_list[0]
    else:
        return np.array(slices_list)


def get_edge_widths(
    edge,
    pos,
    segmented,
    nx_graph_pruned,
    resolution=1,
    offset=4,
    step=30,
    target_length=10,
    bounds=(0, 1),
    order=None,
    logging=False,
):
    target_length = target_length * 2
    slices_list = []
    pixel_list = orient(
        nx_graph_pruned.get_edge_data(*edge)["pixel_list"], pos[edge[0]]
    )
    offset = max(
        offset, step
    )  # avoiding index out of range at start and end of pixel_list
    pixel_indexes = generate_pivot_indexes(
        len(pixel_list), resolution=resolution, offset=offset
    )
    list_of_segments = compute_section_coordinates(
        pixel_list, pixel_indexes, step=step, target_length=target_length + 1
    )
    perp_lines = []

    for i, sect in enumerate(list_of_segments):
        point1 = np.array([sect[0][0], sect[0][1]])
        point2 = np.array([sect[1][0], sect[1][1]])
        perp_lines.append(extract_perp_lines(point1, point2))

    order = validate_interpolation_order(segmented.dtype, order)
    im = segmented
    l = []
    for perp_line in perp_lines:
        pixels = ndi.map_coordinates(
            im, perp_line, prefilter=order > 1, order=order, mode="reflect", cval=0.0
        )
        pixels = np.flip(pixels, axis=1)
        pixels = pixels[int(bounds[0] * target_length) : int(bounds[1] * target_length)]
        pixels = pixels.reshape((1, len(pixels)))
        l.append(pixels)
    slices = np.concatenate(l, axis=0)
    slices_list = [
        max(
            (
                sum(1 for _ in group)
                for value, group in itertools.groupby(pixel_row)
                if value == 0
            ),
            default=0,
        )
        for pixel_row in slices
    ]
    # TODO(FK): Add thickness of the profile here
    #         slices_list.append(max(len(list(y)) for (c,y) in itertools.groupby(np.array(pixels)) if c==1))

    return np.array(slices_list)


def extract_perp_lines(src, dst, linewidth=1):
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)

    length = int(np.ceil(np.hypot(d_row, d_col) + 1))
    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    # we subtract 1 from linewidth to change from pixel-counting
    # (make this line 3 pixels wide) to point distances (the
    # distance between pixel centers)
    col_width = (linewidth - 1) * np.sin(-theta) / 2
    row_width = (linewidth - 1) * np.cos(theta) / 2
    perp_rows = np.stack(
        [
            np.linspace(row_i - row_width, row_i + row_width, linewidth)
            for row_i in line_row
        ]
    )
    perp_cols = np.stack(
        [
            np.linspace(col_i - col_width, col_i + col_width, linewidth)
            for col_i in line_col
        ]
    )
    return np.stack([perp_rows, perp_cols])


def validate_interpolation_order(image_dtype, order):
    """Validate and return spline interpolation's order.

    Parameters
    ----------
    image_dtype : dtype
        Image dtype.
    order : int, optional
        The order of the spline interpolation. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.

    Returns
    -------
    order : int
        if input order is None, returns 0 if image_dtype is bool and 1
        otherwise. Otherwise, image_dtype is checked and input order
        is validated accordingly (order > 0 is not supported for bool
        image dtype)

    """

    if order is None:
        return 0 if image_dtype == bool else 1

    if order < 0 or order > 5:
        raise ValueError("Spline interpolation order has to be in the " "range 0-5.")

    if image_dtype == bool and order != 0:
        raise ValueError(
            "Input image dtype is bool. Interpolation is not defined "
            "with bool data type. Please set order to 0 or explicitely "
            "cast input image to another data type."
        )

    return order


def segment_fluo(
    image,
    thresh=0.5e-7,
    seg_thresh=4.5,
    k_size=40,
    magnif=50,
    binning=2,
    test_plot=False,
):
    k_size = [30, 15][magnif < 50]
    kernel = np.ones((k_size, k_size), np.uint8)
    kernel_2 = np.ones((10, 10), np.uint8)
    smooth_im = cv2.GaussianBlur(image, (5, 5), 0)
    smooth_im = cv2.morphologyEx(smooth_im, cv2.MORPH_OPEN, kernel)
    if magnif < 30:
        im_canny = cv2.Canny(smooth_im, 0, 20)
        smooth_im = cv2.morphologyEx(im_canny, cv2.MORPH_DILATE, kernel)
    _, segmented = cv2.threshold(smooth_im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if magnif > 30:
        segmented = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, np.ones((9, 9)))

    skeletonized = skeletonize(segmented > thresh)
    skeleton = scipy.sparse.dok_matrix(skeletonized)
    nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
    nx_graph_pruned, pos = remove_spurs(nx_graph, pos, threshold=200)
    return (segmented > thresh, nx_graph, pos)


def segment_std(
    frames,
    thresh=0.5e-7,
    seg_thresh=4.5,
    k_size=40,
    magnif=50,
    binning=2,
    test_plot=False,
):
    #     imgs = sorted([path for path in img_address.glob("*/*.ti*")])
    #     frames = []
    #     for i, address in enumerate(img_address):
    #     for i, frame in enumerate(imgs):
    #         if i<framenr:
    #             frame = imageio.imread(self.selection_file[address])
    #             frames.append(frame)
    video_matrix = np.stack(frames, axis=0)
    smooth_im = np.std(video_matrix, axis=0)
    # it seems to be a 64bit image but it has to be 8 or 16 for thresholding (maybe it is in color, but Simon didnt do something with that either)
    smooth_im = cv2.cvtColor(smooth_im, cv2.COLOR_BGR2GRAY)
    smooth_im = cv2.normalize(
        smooth_im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_16U
    )
    #     print(magnif)
    if magnif < 30:
        im_canny = cv2.Canny(smooth_im, 0, 20)
        smooth_im = cv2.morphologyEx(im_canny, cv2.MORPH_DILATE, kernel)
    _, segmented = cv2.threshold(smooth_im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if magnif > 30:
        segmented = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, np.ones((9, 9)))

    skeletonized = skeletonize(segmented > thresh)
    skeleton = scipy.sparse.dok_matrix(skeletonized)
    nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
    nx_graph_pruned, pos = remove_spurs(nx_graph, pos, threshold=200)
    return (segmented > thresh, nx_graph, pos)


def find_thresh_fluo(blurred_image, thresh_guess=40, fold_thresh=0.005):
    histr = cv2.calcHist([blurred_image], [0], None, [256], [0, 256])
    histr_sum = np.sum(histr[0:thresh_guess])
    for i in range(thresh_guess, 1, -1):
        diff = (histr[i] - histr[i - 1]) / histr_sum
        if -1 * diff > fold_thresh:
            break
    return i
