import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import cvlib as cv
import sys
import cv2
import os


def find_max_face(input_image_path, output_image_path):
    """
        Crops face in input_image_path with maximum confidence and writes it to output_image_path
        Adds padding of 15px around face
    """
    image = cv2.imread(input_image_path)
    detection = cv.detect_face(image)
    if detection is not None:
        faces, confidences = detection

        max_conf = max(confidences)
        max_face = faces[confidences.index(max_conf)]
        face = max_face

        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]
        cv2.imwrite(
            output_image_path, image[startY - 15 : endY + 15, startX - 15 : endX + 15]
        )


if __name__ == "__main__":
    find_max_face(sys.argv[1], "face.jpg")
    cv2.destroyAllWindows()
