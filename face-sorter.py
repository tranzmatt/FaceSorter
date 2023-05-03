import os
import shutil
import face_recognition
import torch
import cv2
import argparse
import re

def load_known_faces(known_faces_folder):
    # Load the known face encodings and names from the known_faces_folder
    known_face_encodings = []
    known_face_names = []

    for file in os.listdir(known_faces_folder):
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            image_path = os.path.join(known_faces_folder, file)
            face_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(face_image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(os.path.splitext(file)[0])

    return known_face_encodings, known_face_names


def get_model_path():
    # Get the path to the face recognition model
    return os.path.join(os.path.dirname(__file__), 'models/face_recognition_model')


def detect_faces_in_image(image_path, face_recognition_model, confidence_threshold=0.6):
    # Load the image
    image = face_recognition.load_image_file(image_path)

    # Detect faces in the image
    face_locations = face_recognition.api.batch_detect(image, model=face_recognition_model)
    face_encodings = face_recognition.face_encodings(image, face_locations, model='small')

    # Filter out face encodings with distance greater than confidence_threshold
    face_locations = [face_locations[i] for i in range(len(face_encodings)) if face_recognition.face_distance([face_encodings[i]], known_face_encodings)[0] < confidence_threshold]
    face_encodings = [face_encodings[i] for i in range(len(face_encodings)) if face_recognition.face_distance([face_encodings[i]], known_face_encodings)[0] < confidence_threshold]

    # Return the face locations and encodings
    return face_locations, face_encodings


def find_matching_known_face(face_encoding, known_face_encodings, known_face_names, tolerance=0.6):
    # Check if the face matches any of the known faces
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    matches = face_distances < tolerance
    for i, match in enumerate(matches):
        if match:
            return known_face_names[i]
    return None


def process_key_frame(frame, output_folder, face_recognition_model, known_face_encodings, known_face_names, confidence_threshold=0.6):
    # Detect faces in the key frame
    _, _, height, width = cv2.imencode('.jpg', frame).shape
    image = frame[:, :, ::-1]  # Convert BGR (OpenCV) to RGB (face_recognition)
    face_locations, face_encodings = detect_faces_in_image(image, face_recognition_model, confidence_threshold=confidence_threshold)

    # If no faces are detected, skip this key frame
    if len(face_encodings) == 0:
        return

    # Loop through each face in the key frame and compare it to the known faces
    for i, face_encoding in enumerate(face_encodings):
        # Check if the face matches any of the known faces
        matching_known_face_name = find_matching_known_face(face_encoding, known_face_encodings, known_face_names, tolerance=0.6)

        if matching_known_face_name is not None:
            # If the face matches a known face, put it in the corresponding folder
            face_folder = os.path.join(output_folder, matching_known_face_name)
        else:
            # If the face doesn't match any known face, put it in a new folder
            face_folder = os.path.join(output_folder, f'unknown_face_{i+1}')

        if not os.path.exists(face_folder):
            os.makedirs(face_folder)

        # Save the face as an image in the corresponding folder
        top, right, bottom, left = face_locations[i]
        face_image = frame[top:bottom, left:right]
        face_filename = os.path.join(face_folder, f'{len(os.listdir(face_folder)) + 1}.jpg')
        cv2.imwrite(face_filename, face_image)


def process_key_frame(frame, key_frame_folder, face_recognition_model, known_face_encodings, known_face_names, confidence_threshold=0.6):
    # Detect faces in the key frame
    face_locations, face_encodings = detect_faces_in_image(frame, face_recognition_model, confidence_threshold=confidence_threshold)

    # If no faces are detected, skip this key frame
    if len(face_encodings) == 0:
        return

    # Create a folder for this key frame's faces if it doesn't exist
    if not os.path.exists(key_frame_folder):
        os.makedirs(key_frame_folder)

    # Loop through each face in the key frame and compare it to the known faces
    for i, face_encoding in enumerate(face_encodings):
        # Check if the face matches any of the known faces
        matching_known_face_name = find_matching_known_face(face_encoding, known_face_encodings, known_face_names, tolerance=0.6)

        if matching_known_face_name is not None:
            # If the face matches a known face, put it in the corresponding folder
            face_folder = os.path.join(key_frame_folder, matching_known_face_name)
        else:
            # If the face doesn't match any known face, put it in a new folder
            face_folder = os.path.join(key_frame_folder, f'face{i+1}')
            os.makedirs(face_folder)


def process_video(input_video, output_folder, known_faces_folder, timeout=10, confidence_threshold=0.8):
    # Load the known faces
    known_face_encodings, known_face_names = load_known_faces(known_faces_folder)

    # Load the face recognition model
    face_recognition_model = face_recognition.api.load_model(get_model_path())

    # Check if the input is a file or a stream
    if os.path.isfile(input_video):
        # Input is a file, process as video file
        cap = cv2.VideoCapture(input_video)
    else:
        # Input is a stream, process as video stream
        cap = cv2.VideoCapture(input_video)

    # Set a timeout for the video capture function
    cap.set(cv2.CAP_PROP_FPS, timeout)

    # Extract key frames and detect faces in each key frame
    key_frame_interval = 10  # Extract a key frame every 10 frames
    key_frame_count = 0
    while cap.isOpened():
        # Read the next frame from the video file or stream
        ret, frame = cap.read()
        if not ret:
            break

        # Extract a key frame every key_frame_interval frames
        if key_frame_count % key_frame_interval == 0:
            key_frame_folder = os.path.join(output_folder, f'key_frame_{key_frame_count}')
            process_key_frame(frame, key_frame_folder, face_recognition_model, known_face_encodings, known_face_names, confidence_threshold=confidence_threshold)

        key_frame_count += 1

    # Release the video file or stream
    cap.release()


def process_image(input_image, output_folder, known_faces_folder, confidence_threshold=0.6):
    # Load the known faces
    known_face_encodings, known_face_names = load_known_faces(known_faces_folder)

    # Load the face recognition model
    face_recognition_model = face_recognition.api.load_model(get_model_path())

    # Detect faces in the input image
    key_frame_folder = os.path.join(output_folder, os.path.splitext(os.path.basename(input_image))[0])
    process_key_frame(input_image, key_frame_folder, face_recognition_model, known_face_encodings, known_face_names, confidence_threshold=confidence_threshold)


def is_video_url(url):
    # Check if the input is a video stream URL (RTSP, UDP, HTTP, or HTTPS)
    video_url_regex = re.compile('^(rtsp|udp|http|https)://.+')
    return video_url_regex.match(url) is not None


def process_input(input_source, output_folder, known_faces_folder, timeout=10, confidence_threshold=0.8):
    if os.path.isdir(input_source):
        for file in os.listdir(input_source):
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                image_path = os.path.join(input_source, file)
                process_image(image_path, output_folder, known_faces_folder, confidence_threshold=confidence_threshold)
    elif os.path.isfile(input_source) or is_video_url(input_source):
        process_video(input_source, output_folder, known_faces_folder, timeout=timeout, confidence_threshold=confidence_threshold)
    else:
        print("Error: Invalid input source. Please provide a video file, video stream URL, or folder of images.")
        exit(1)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sort faces in a video file, video stream, or a folder of images.')
    parser.add_argument('-i', '--input', dest='input_source', required=True, help='Path to input source: video file, RTSP URL, or folder of images')
    parser.add_argument('-o', '--output', dest='output_folder', required=True, help='Path to output folder')
    parser.add_argument('-k', '--known', dest='known_faces_folder', required=True, help='Path to folder containing known face images')
    parser.add_argument('-t', '--timeout', dest='timeout', type=int, default=10, help='Timeout for video capture function (only applicable for video inputs)')
    parser.add_argument('-c', '--confidence', dest='confidence_threshold', type=float, default=0.8, help='Threshold for face recognition confidence')
    args = parser.parse_args()

    # Process the input source: video file, video stream, or folder of images
    process_input(args.input_source, args.output_folder, args.known_faces_folder, timeout=args.timeout, confidence_threshold=args.confidence_threshold)

