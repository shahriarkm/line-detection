import cv2
import numpy as np


def find_lane_pixels(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2 :, :], axis=0)
    # print(binary_warped[binary_warped.shape[0] // 2 :, :])
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 30
    minpix = 40

    window_height = binary_warped.shape[0] // nwindows
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # print(nonzerox)
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(
            out_img,
            (int(win_xleft_low), int(win_y_low)),
            (int(win_xleft_high), int(win_y_high)),
            (0, 255, 0),
            2,
        )
        cv2.rectangle(
            out_img,
            (int(win_xright_low), int(win_y_low)),
            (int(win_xright_high), int(win_y_high)),
            (0, 255, 0),
            2,
        )

        good_left_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xleft_low)
            & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xright_low)
            & (nonzerox < win_xright_high)
        ).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.mean(nonzerox[good_left_inds])
        if len(good_right_inds) > minpix:
            rightx_current = np.mean(nonzerox[good_right_inds])

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except:
        pass

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img, leftx_base, rightx_base


def fit_polynomial(binary_warped):
    leftx, lefty, rightx, righty, out_img, leftx_base, rightx_base = find_lane_pixels(
        binary_warped
    )

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # print(ploty)
    try:
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    except TypeError:
        print("The function failed to fit a line!")
        left_fitx = 1 * ploty**2 + 1 * ploty
        right_fitx = 1 * ploty**2 + 1 * ploty

    # print(right_fitx)
    # print(left_fitx)

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    # print(left_fitx)

    unwarp = np.zeros_like(out_img)
    curveL = np.column_stack((left_fitx.astype(np.int32), ploty.astype(np.int32)))
    cv2.polylines(unwarp, [curveL], False, (255, 0, 0), 5)
    curveR = np.column_stack((right_fitx.astype(np.int32), ploty.astype(np.int32)))
    cv2.polylines(unwarp, [curveR], False, (255, 0, 0), 5)
    x = binary_warped.shape[0] - 1
    # poly = np.zeros_like(out_img)
    # print(curveR)
    # print(curveR[-1, 0])
    # print(curveR[0, 0])
    # points = np.array(
    #     [[curveL[0, 0], 0], [curveR[0, 0], 0], [curveR[-1, 0], -1], [curveL[-1, 0], -1]]
    # )
    # poly[points] = (255, 255, 255)
    # cv2.fillPoly(poly, points, (0, 255, 0), 5)

    finalFunc = (np.array(right_fit) + np.array(left_fit)) / 2
    # print(finalFunc)
    # out_img[right_fitx, ploty] = [0, 255, 255]
    # out_img[left_fitx, ploty] = [0, 255, 255]
    # print(left_fitx.shape, ploty.shape)
    # print("\n\n\n\n\n\n\n\n\n################\n\n\n\\n\n\n\n\n\n\n\n\n\n")
    offset = (
        (
            (binary_warped.shape[0] // 2 - leftx_base)
            - (rightx_base - binary_warped.shape[0] // 2)
        )
        * 3.7
        / (1280 / 4)
        / 10
    )

    return unwarp, finalFunc, offset


def warp(img, warp=True):
    sizeWarp = (1280 // 4, 720 // 4)
    sizeUnwarp = (1280 // 2, 720 // 2)

    src = np.float32(
        [
            [590 // 2, 450 // 2],
            [690 // 2, 450 // 2],
            [1080 // 2, 700 // 2],
            [200 // 2, 700 // 2],
        ]
    )
    dst = np.float32(
        [
            [250 // 4, 0 // 4],
            [1030 // 4, 0 // 4],
            [1030 // 4, 720 // 4],
            [250 // 4, 720 // 4],
        ]
    )
    if warp:
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, matrix, sizeWarp, flags=cv2.INTER_LINEAR)
    else:
        matrix = cv2.getPerspectiveTransform(dst, src)
        return cv2.warpPerspective(img, matrix, sizeUnwarp, flags=cv2.INTER_LINEAR)


def abs_sobel_thresh(gray, orientx=True, k=3, thresh=(0, 255)):
    if orientx:
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k))
    else:
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return gradmag


def dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= (thresh[0])) & (absgraddir <= thresh[1])] = 255
    return binary_output
