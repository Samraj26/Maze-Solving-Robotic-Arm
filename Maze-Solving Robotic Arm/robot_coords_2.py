import matplotlib.pyplot as plt
import cv2
import numpy as np
import threading
import pickle 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def detect():
    cap = cv2.VideoCapture(2)
    with open("/home/anirudh/Desktop/ASU/Robotics/Lab4/polynomial_features.pkl", "rb") as f:
        poly_loaded = pickle.load(f)
    with open("/home/anirudh/Desktop/ASU/Robotics/Lab4/polynomial_regression_model.pkl", "rb") as f:
        model_loaded = pickle.load(f)

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
            test_point_poly = poly_loaded.transform(centroids)
            centroids = model_loaded.predict(test_point_poly) 
            robot_coords = np.concatenate([centroids, np.ones((2, 1))], axis=1)
            break
    cv2.destroyAllWindows() 
    cv2.waitKey(1)

    return robot_coords
    
def trial():
    cap = cv2.VideoCapture(2)
    with open("/home/anirudh/Desktop/ASU/Robotics/Lab4/polynomial_features.pkl", "rb") as f:
        poly_loaded = pickle.load(f)
    with open("/home/anirudh/Desktop/ASU/Robotics/Lab4/polynomial_regression_model.pkl", "rb") as f:
        model_loaded = pickle.load(f)
        
    centroids = np.array([[475,370],[229,118]])
    test_point_poly = poly_loaded.transform(centroids)
    centroids = model_loaded.predict(test_point_poly) 
    robot_coords = np.concatenate([centroids, np.ones((2, 1))], axis=1)
    return robot_coords

if __name__=="__main__":
    robot_coords = trial()#detect()
