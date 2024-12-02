import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Parameters
IMG_DIR = 'data/train'  # Directory containing images

def preprocess_image(image):
    """ Convert to grayscale and apply Gaussian blur. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def segment_trees(image):
    """ Segment the trees in the image using watershed algorithm. """
    # Apply preprocessing
    blurred = preprocess_image(image)

    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find sure foreground area
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Find unknown region
    unknown = cv2.subtract(dilated, np.uint8(sure_fg))

    # Marker labelling
    _, markers = cv2.connectedComponents(np.uint8(sure_fg))

    # Add one to all labels to distinguish sure regions from unknown
    markers = markers + 1
    markers[unknown == 255] = 0  # Mark unknown region as 0

    # Apply watershed algorithm
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Mark boundaries in red

    return markers, image

def visualize_segmentation(original_img, segmented_img):
    """ Display the original image and the segmented image side by side. """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(segmented_img)
    ax[1].set_title("Segmented Image")
    ax[1].axis("off")

    plt.show()

def count_trees(segmented_img):
    """ Count the number of unique labels in the segmented image. """
    num_trees = np.unique(segmented_img).shape[0] - 1  # Exclude the background
    return num_trees

def main():
    # Process each image in the IMG_DIR
    for img_name in os.listdir(IMG_DIR):
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Segment trees in the image
        markers, segmented_img = segment_trees(img)

        # Visualize original and segmented images
        visualize_segmentation(img, segmented_img)

        # Count trees
        tree_count = count_trees(markers)
        print(f"Estimated number of trees in {img_name}: {tree_count}")

if __name__ == "__main__":
    main()
