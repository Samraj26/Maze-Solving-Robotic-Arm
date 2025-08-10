import matplotlib.pyplot as plt
import cv2
import numpy as np
import threading

def detect():
    cap = cv2.VideoCapture(2)
    mat = np.load("/home/anirudh/Desktop/ASU/Robotics/Lab4/transformation_matrix.npy")

    while True:
        success, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            cv2.imshow('Detected Markers', image)
            
        if len(corners)==2:
            print(corners)
            cap.release()
            corners = np.array(corners)
            centroids  = corners.mean(axis=2)
            centroids = centroids.reshape(2, 2)
            centroids = np.concatenate([centroids, np.ones((2, 1))], axis=1)
            print(centroids)
            robot_coords = (mat@centroids.T).T
            print(robot_coords)
            break
    cv2.destroyAllWindows() 
    cv2.waitKey(1)

    return robot_coords
    
def trial():
	mat = np.load("/home/anirudh/Desktop/ASU/Robotics/Lab4/transformation_matrix.npy")
	centroids = np.array([[475, 370   , 1. ],[229, 118 ,  1. ]])
	robot_coords = (mat@centroids.T).T
	return robot_coords

if __name__=="__main__":
    robot_coords = trial()#detect()
    #print(robot_coords)
