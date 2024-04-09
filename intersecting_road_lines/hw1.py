import cv2
import numpy as np
import logging

def main():
    logging.basicConfig(level="INFO")
    image = cv2.imread('road1.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, _ = gray.shape[:2] # height for ROI

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 200, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    parallel_lines = []
    for line1 in lines:
        rho1, theta1 = line1[0]
        a1 = np.cos(theta1)
        b1 = np.sin(theta1)
        x01 = a1 * rho1
        y01 = b1 * rho1
        x11 = int(x01 + 1000 * (-b1))
        y11 = int(y01 + 1000 * (a1))
        x21 = int(x01 - 1000 * (-b1))
        y21 = int(y01 - 1000 * (a1))
        for line2 in lines:
            rho2, theta2 = line2[0]
            a2 = np.cos(theta2)
            b2 = np.sin(theta2)
            x02 = a2 * rho2
            y02 = b2 * rho2
            x12 = int(x02 + 1000 * (-b2))
            y12 = int(y02 + 1000 * (a2))
            x22 = int(x02 - 1000 * (-b2))
            y22 = int(y02 - 1000 * (a2))
            # Parallelism test
            angle_diff = np.abs(theta1 - theta2)
            if angle_diff > np.pi / 2 - np.pi / 18 and angle_diff < np.pi / 2 + np.pi / 18:
                parallel_lines.append([(x11, y11), (x21, y21)])
                parallel_lines.append([(x12, y12), (x22, y22)])
                break

    logging.info(f"parallel lines: {parallel_lines}")
    parallel_lines = parallel_lines[:2] # these lines are enough
    logging.info(f"parallel lines after: {parallel_lines}")

    intersection = np.zeros(2)
    count = 0
    for i in range(len(parallel_lines)):
        for j in range(i+1, len(parallel_lines)):
            x1, y1 = parallel_lines[i][0]
            x2, y2 = parallel_lines[i][1]
            x3, y3 = parallel_lines[j][0]
            x4, y4 = parallel_lines[j][1]
            
            denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denominator != 0:
                intersection[0] += ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
                intersection[1] += ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
                count += 1

    # Normalize
    if count != 0:
        intersection /= count
    intersection = intersection.astype(int)
    logging.info(f"Intersection point: {intersection}")

    # Find ROI (bottom 70% of image)
    roi_bottom = int(height * 0.3)

    for i in range(len(parallel_lines)):
        x1, y1 = parallel_lines[i][0]
        x2, y2 = parallel_lines[i][1]
        parallel_lines[i][0] = (x1, y1 - roi_bottom)
        parallel_lines[i][1] = (x2, y2 - roi_bottom)

    # Draw results
    for line in parallel_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        if y1 < 0:
            x1 = int(x1 - (y1 / (y2 - y1)) * (x2 - x1))
            y1 = 0
        if y2 < 0:
            x2 = int(x2 - (y2 / (y2 - y1)) * (x2 - x1))
            y2 = 0
        cv2.line(image, (x1, y1 + roi_bottom), (x2, y2 + roi_bottom), (0, 0, 255), 4)
    cv2.circle(image, (int(intersection[0]), int(intersection[1])), 8, (255,165,0), -1)


    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ ==  "__main__":
    main()
