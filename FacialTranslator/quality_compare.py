from __future__ import annotations
import socket
import cv2
import math
import numpy as np
from argparse import ArgumentParser

from pylivelinkface import PyLiveLinkFace, FaceBlendShape
import mediapipe as mp
from mediapipe.python.solutions import face_mesh, drawing_utils, drawing_styles
from face_geometry import (
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)

def calculate_rotation(face_landmarks, pcf: PCF, image_shape):
    landmarks = np.array(
        [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark[:468]]
    )
    landmarks = landmarks.T

    metric_landmarks, pose_transform_mat = get_metric_landmarks(
        landmarks.copy(), pcf
    )

    return pose_transform_mat, metric_landmarks

def process_image(image):
        pcf = PCF(
            near=1,
            far=10000,
            frame_height=480,
            frame_width=640,
            fy=640,
        )
        
        face_mesh_ref = face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh_ref.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        face_image_3d = None
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                pose_transform_mat, metric_landmarks = calculate_rotation(face_landmarks, pcf, image.shape)
                return metric_landmarks
                
if __name__ == "__main__":
    source_img = cv2.imread('images/source_image5.jpg')
    target_img = cv2.imread('images/solution_image5.jpg')
    source_metrics = process_image(source_img)
    target_metrics = process_image(target_img)
    
    distances = [math.dist(m1, m2) * math.dist(m1, m2) for m1,m2 in zip(source_metrics.T, target_metrics.T)]
    print(np.sum(distances))

