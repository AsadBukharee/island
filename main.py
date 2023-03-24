# get_island(img) is the only usefull function here.
# as me any questions : asadbukharee@gmail.com

import cv2
import numpy as np
# you don't need skimage or scipy to get island only.
from skimage.segmentation import active_contour
from scipy.spatial.distance import directed_hausdorff
import os
# if you don't want to see the output images, set it to False
DEBUG = True
directory = 'images'
dataset = []
# Thats it.!

def threshold_image(img):
    win_name = 'Threshold Image'
    # Create a window and trackbars
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Low Threshold', win_name, 0, 255, lambda x: None)
    cv2.createTrackbar('High Threshold', win_name, 255, 255, lambda x: None)
    cv2.createTrackbar('Blurr', win_name, 3, 14, lambda x: None)

    while True:
        # Get the current trackbar values
        low_threshold = cv2.getTrackbarPos('Low Threshold', win_name)
        high_threshold = cv2.getTrackbarPos('High Threshold', win_name)
        ksize = cv2.getTrackbarPos('Blurr', win_name)
        ksize = max(3, 2 * ksize + 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply the threshold to the image
        blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        thresholded = cv2.threshold(blur, low_threshold, high_threshold, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Show the thresholded image
        cv2.imshow(win_name, thresholded)

        # Wait for a key event
        key = cv2.waitKey(1) & 0xFF

        # Check if the 'q' key was pressed to exit the loop
        if key == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()


def get_island(img):
    ksize = 7
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply the threshold to the image
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Loop through the contours and find the first contour with area greater than a threshold
    threshold_area = 10000  # Adjust this threshold depending on your image and the size of the main island
    mask = None
    for c in contours:
        if cv2.contourArea(c) > threshold_area:
            # Draw the contour of the main island on a new image
            mask = np.zeros_like(img)
            cv2.drawContours(mask, [c], 0, (0, 0, 255), -1)
            break
    island = cv2.bitwise_and(img, img, mask=mask[:, :, 2])
    return mask, island


def show(name=None, img=None):
    try:
        window_name = 'Image Window' if name is None else name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"{e}")


def add_to_dataset(file):
    global dataset
    global DEBUG
    path = os.path.join(directory, file)

    # Step 1: Extract contour from OpenStreetMap image
    img = cv2.imread(path)
    mask, island = get_island(img)
    if DEBUG:
        show(name='mask', img=mask)
        show(name='island', img=island)
    # threshold_image(img)
    edges = cv2.Canny(mask, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0][:, 0, :]
    img = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    if DEBUG:
        show(name='snake', img=img)
    # Step 2: Create active contour model
    s = np.linspace(0, 2 * np.pi, len(contour))
    init = np.column_stack((contour[:, 0], contour[:, 1]))
    snake = active_contour(edges, init, alpha=0.015, beta=10, gamma=0.001)

    # Step 3: Create dataset for the island

    for i in range(36):
        angle = i * 10
        rot_mat = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
        snake_homog = np.column_stack((snake, np.ones(len(snake))))
        rotated_snake_homog = np.dot(rot_mat, snake_homog.T).T
        rotated_snake = rotated_snake_homog[:, :2]
        dataset.append((rotated_snake[:, :2], file.split('.')[0]))  # file is the image name, in fact the island
    if DEBUG:
        print(dataset)


# you can use following code as per you need. you can test and modify
# the goal of creating dataset is achieved.
# classify is not useful here.
def classify(file):
    # Step 4: Classify new image using dataset
    path = os.path.join(directory, file)
    new_img = cv2.imread(path)
    new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    new_edges = cv2.Canny(new_gray, 100, 200)
    new_contours, _ = cv2.findContours(new_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    new_contour = new_contours[0][:, 0, :]
    new_s = np.linspace(0, 2 * np.pi, len(new_contour))
    new_init = np.column_stack((new_contour[:, 0], new_contour[:, 1]))
    new_snake = active_contour(new_edges, new_init, alpha=0.015, beta=10, gamma=0.001)
    min_dist = float('inf')
    min_label = None
    for data in dataset:
        dist = directed_hausdorff(data[0], new_snake)[0]
        if dist < min_dist:
            min_dist = dist
            min_label = data[1]
    print('The island is:', min_label)
    return min_label


def read_files(path):
    files = os.listdir(path)
    return [file for file in files if not file.startswith(".")]


if __name__ == '__main__':
    files = read_files(directory)
    print(files)
    for file_name in files:
        add_to_dataset(file_name)

    print("NOW WE TEST IT FOR CLASSIFICATION")
    classify(files[0])  # you can pass any image here,
