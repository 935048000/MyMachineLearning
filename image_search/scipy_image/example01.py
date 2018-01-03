import cv2
import scipy as sp

def imagess(image1,image2):
    img1 = cv2.imread (image1, 0)
    img2 = cv2.imread (image2, 0)

    # Initiate SIFT detector
    # sift = cv2.SIFT ()
    sift = cv2.xfeatures2d.SIFT_create ()


    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute (img1, None)
    kp2, des2 = sift.detectAndCompute (img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict (algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict (checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher (index_params, search_params)
    matches = flann.knnMatch (des1, des2, k=2)

    print('matches...', len (matches))

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append (m)
    print('good', len (good))

    # #####################################
    # visualization
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    view = sp.zeros ((max (h1, h2), w1 + w2, 3), sp.uint8)
    view[:h1, :w1, 0] = img1
    view[:h2, w1:, 0] = img2
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]

    for m in good:
        # draw the keypoints
        # print m.queryIdx, m.trainIdx, m.distance
        color = tuple ([sp.random.randint (0, 255) for _ in xrange (3)])
        # print 'kp1,kp2',kp1,kp2
        cv2.line (view, (int (kp1[m.queryIdx].pt[0]), int (kp1[m.queryIdx].pt[1])),
                  (int (kp2[m.trainIdx].pt[0] + w1), int (kp2[m.trainIdx].pt[1])), color)

    cv2.imshow ("view", view)
    cv2.waitKey ()

if __name__ == '__main__':
    image1 = './test01/1.JPG'
    image2 = './test01/2.JPG'

    imagess(image1,image2)
