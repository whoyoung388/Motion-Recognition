import os
import pickle

import cv2
import numpy as np

def draw_contour(mhi, dst_image, action, probability):
    """
    Args:
        mhi (np.uint8): motion history image
        dst_image (np.uint8): destination where contour to be draw on
        action (str): predict motion name
        probability (float): returned prob from classifier

    Return:
        image with contour drawn on it
    """
    if mhi.sum() < 1000:
        return dst_image
    # find the latest motion
    src = (mhi == mhi.max()).astype(np.uint8)
    # blur the image with weighting
    kernel = np.ones((20, 20))/10
    blured = cv2.filter2D(src, -1, kernel)
    # binary the blured image to find contoure
    binary = ((blured != 0) * 255).astype(np.uint8)

    # the center
    M = cv2.moments(binary)
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except ZeroDivisionError:
        return dst_image

    # rectangle coordinate
    boundX = [i for i, val in enumerate(binary.argmax(axis=0)) if val]
    boundY = [i for i, val in enumerate(binary.argmax(axis=1)) if val]
    if len(boundX) == 0:
        return dst_image
    if len(boundY) == 0:
        return dst_image
    topX, bottomX = boundX[0], boundX[-1]
    topY, bottomY = boundY[0], boundY[-1]

    # draw on dst_image
    image = dst_image.copy()
    # if probability > 0.1:
    cv2.putText(image, '{0}'.format(action), (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, 0, 1)
    cv2.putText(image, '{0:.1f}%'.format(probability*100), (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, 0, 1)
    cv2.rectangle(image, (topX, topY), (bottomX, bottomY), 0, 1)
    cv2.circle(image, (cX, cY), 0, 0, 3)
    return image


def mhi_matching(clf, mhi):
    """
    Args:
        clf (object): classifier, default will use SVM loaded
    """
    assert mhi.shape == (120, 160), "wrong size of MHI, expect (160, 120), received {}".format(mhi.shape)
    motion_probs = clf.predict_proba([mhi.reshape(-1,),])[0]
    most_likeli_ind = motion_probs.argmax() // 4
    return most_likeli_ind, motion_probs[most_likeli_ind]


def compute_diff(image1, image2):
    """
    Reference from
    https://stackoverflow.com/questions/35777830/fast-absolute-difference-of-two-uint8-arrays

    Args:
        image1 (np.uint8) frame at t from input video
        image2 (np.uint8) frame at t+1 from input video

    Return:
        diff image (np.uint8)
    """
    direct_diff = image1 - image2
    correction = np.uint8(image1 < image2) * 254 + 1
    correct_diff = direct_diff * correction
    return correct_diff


def main(video_path, tau=15, threshold=50, auto_play=False):
    categories = ['running', 'jogging', 'walking',
                'handclapping', 'handwaving', 'boxing']

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracked_frame = frame.copy()
    mhi = np.zeros(frame.shape)
    mhi_norm = mhi.copy().astype(np.uint8)

    # Start go through the video
    while ret:
        cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('demo', 321, 481)
        enlarge_f = cv2.resize(tracked_frame, (320, 240), interpolation=cv2.INTER_NEAREST)
        # enlarge_f = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
        enlarge_m = cv2.resize(mhi_norm, (320, 240), interpolation=cv2.INTER_NEAREST)
        combine = cv2.vconcat((enlarge_f, enlarge_m))
        cv2.imshow('demo', combine)
        cv2.waitKey(auto_play)
        ret, frame_next = cap.read()
        if ret:
            frame_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
            diff = compute_diff(frame, frame_next)
            mhi[diff > threshold] = tau
            mhi[(diff <= threshold)*(mhi > 0)] -= 1
            mhi_norm = (mhi.copy() * 255 / tau).astype(np.uint8)
            frame = frame_next.copy()

            category_ind, prob = mhi_matching(classifier, mhi_norm)
            # if prob > 0.1:
            action = categories[category_ind]
            tracked_frame = draw_contour(mhi, frame.copy(), action, prob)
        else:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    sample_video = './sample/person15_running_d1_uncomp.avi'
    # Load trained model
    if os.path.isfile('./SVM_general_model.sav'):
        classifier = pickle.load(open('SVM_general_model.sav', 'rb'))
        main(sample_video, auto_play=False)
    else:
        print "Training Model doesn't exist."
