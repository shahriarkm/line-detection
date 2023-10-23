import cv2
import numpy as np
import manip_functions as f


def measure_curvature_pixels(ploty, left_fit, right_fit):
    y_eval = np.max(ploty)
    left_curverad = (
        (1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5
    ) / np.absolute(2 * left_fit[0])
    right_curverad = (
        (1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5
    ) / np.absolute(2 * right_fit[0])

    return left_curverad, right_curverad


########################################################


cap = cv2.VideoCapture("H:/auriga/cv/Lane Detection Project/project_video.mp4")

ksize = 15

while cap.isOpened():
    # capturing vid
    _, frame_cp = cap.read()
    try:
        frame_cp = cv2.resize(
            frame_cp, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
        )
    except:
        break
    frame = np.copy(frame_cp)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = f.warp(gray)

    gradx = f.abs_sobel_thresh(gray, 1, ksize, thresh=(20, 100))
    s = hls[:, :, 2]
    s = f.warp(s)
    hlSBin = np.zeros_like(s)
    hlSBin[(s > 169) * (s <= 255)] = 1
    sobelS = np.zeros_like(s)
    sobelS[(hlSBin == 1) | (gradx == 1)] = 255

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    v = f.warp(v)
    vBin = np.zeros_like(v)
    vBin[(v > 190)] = 255

    # cv2.imshow("v", vBin)

    # hist = np.zeros_like(gray)
    # hist[:, 640:] = vBin[:, 640:]
    # hist[:, :640] = HsvGBin[:, :640]

    final = np.zeros_like(v)

    final[(sobelS == 255) | (vBin == 255)] = 255

    final[:, : v.shape[1] // 2] = sobelS[:, : v.shape[1] // 2]
    # final[(dir_binary == 255) * (sobelS == 1)] = 255
    final = f.fit_polynomial(final)
    cv2.imshow("f", final)

    ##################################################

    ##################################################
    # cv2.imshow("x", gradx)
    # dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0, 0.4))
    # grady = abs_sobel_thresh(gray, 0, ksize, thresh=(20, 100))
    # # cv2.imshow("y", grady)

    # mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(20, 100))
    # # cv2.imshow("mag", mag_binary)

    # cv2.imshow("dir", dir_binary)

    # combined = np.zeros_like(dir_binary)
    # combined[
    #     ((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))
    # ] = 1
    # combined = warp(combined)
    # cv2.imshow("img", combined)
    ##############
    # img = warp(frame2)
    # img = dir_threshold(frame2)

    # thresholding

    ################################
    # IMPORTANT

    # H = hls[:, :, 0]
    # hBin = np.zeros_like(H)
    # hBin[(H < 25) * (H > 18)] = 255
    # cv2.imshow("h", hBin)

    ##################################

    # canny = cv2.Canny(gray, 220, 150)

    # cv2.imshow("r", warp(rBin))
    # b = frame[:, :, 2]
    # bBin = np.zeros_like(gray)
    # bBin[(b > 110)] = 255
    # cv2.imshow("b", bBin)

    # r = frame[:, :, 0]
    # rBin = np.zeros_like(gray)
    # rBin[(r <= 255) * (r > 180)] = 255
    # cv2.imshow("r", rBin)

    # b = frame[:, :, 2]
    # bBin = np.zeros_like(gray)
    # bBin[(b < 85) * (b > 80)] = 255
    # cv2.imshow("b", bBin)

    # v2 = hsv[:, :, 2]
    # v2Bin = np.zeros_like(v2)
    # v2Bin[(v2 > 160) * (v2 < 162)] = 255
    # cv2.imshow("v", v2Bin)

    #######################################################

    # """Left line"""

    # Hsv = hsv[:, :, 0]
    # HsvBin = np.zeros_like(Hsv)
    # HsvBin[(Hsv > 19) * (Hsv < 26)] = 255
    # g = frame[:, :, 1]
    # gBin = np.zeros_like(gray)
    # gBin[(g > 185)] = 255
    # # cv2.imshow("g", gBin)
    # HsvGBin = np.zeros_like(gray)
    # HsvGBin[(gBin == 255) + (HsvGBin == 255) + (hlSBin == 255)] = 255

    # """Right line"""

    # cv2.imshow("hist", warp(hist))

    ##########################################################

    # hSv = hsv[:, :, 1]
    # hSvBin = np.zeros_like(hSv)
    # hSvBin[(hSv > 85)] = 255
    # mainBin = np.zeros_like(gray)
    # mainBin[(rBin == 255) * (sBin == 255)] = 255

    #########################3
    # # # testing for threshold
    # Hls = hls[:, :, 0]
    # cv2.imshow("H", Hls)

    # l = hls[:, :, 1]
    # cv2.imshow("L", l)

    # hlS = hls[:, :, 2]
    # cv2.imshow("S", hlS)

    # Hsv = hsv[:, :, 0]
    # cv2.imshow("H", Hsv)

    # hSv = hsv[:, :, 1]
    # cv2.imshow("S", hSv)

    # v = hsv[:, :, 2]
    # cv2.imshow("V", v)

    # r = frame[:, :, 0]
    # cv2.imshow("R", r)

    # g = frame[:, :, 1]
    # cv2.imshow("G", g)

    # b = frame[:, :, 2]
    # cv2.imshow("B", b)

    #####################################

    # print(np.shape(sBin))
    # cv2.resize(sBin, (144, 256), interpolation=cv2.INTER_AREA)
    # cv2.imshow("s", warp(sBin))

    # cv2.imshow("main", mainBin)

    # cv2.imshow("hSv", hSvBin)
    # cv2.imshow("v", warp(vBin))
    # cv2.imshow("mag", warp(mbinary))

    y = np.array(
        [
            [590 // 2, 450 // 2],
            [690 // 2, 450 // 2],
            [1080 // 2, 700 // 2],
            [200 // 2, 700 // 2],
        ]
    )
    # cv2.imshow("canny", warp(canny))
    # cv2.imshow("s", s)

    # y = np.array([[595, 452], [690, 452], [1106, 681], [300, 681]])
    # frame = cv2.polylines(frame, np.int32([y]), True, 2)
    cv2.imshow("frame", frame)
    # cv2.imshow("title", frame)

    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break
    elif ord("e"):
        continue

cap.release()
cv2.destroyAllWindows()
