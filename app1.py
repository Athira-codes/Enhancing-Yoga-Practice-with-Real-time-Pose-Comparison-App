# # Enhancing Yoga Practice with Real-time Pose Comparison App


# # Introduction:

# # Many individuals practice yoga by following instructional videos on platforms like YouTube.
# # However, without real-time feedback, it's challenging to ensure accurate pose alignment.
# # Our solution addresses this issue by providing instant feedback on pose correctness, improving the quality of yoga practice.

# # Key Concept:

# # Our app utilizes computer vision technology to compare the user's yoga pose with the instructor's pose in real-time.
# # By analyzing the alignment of key body landmarks, the app provides immediate feedback on pose accuracy.
# # Users receive visual cues indicating whether their pose matches the instructor's, helping them make necessary adjustments for better alignment.



# # Benefits:

# # Enhanced Learning Experience:
# # Users receive personalized guidance on pose alignment, enhancing their understanding and execution of yoga poses.
# # Real-time feedback enables users to correct posture mistakes immediately, leading to a more effective and rewarding practice.
# # Cost-effective Alternative:
# # Traditional yoga classes or private sessions can be expensive, making them inaccessible to many individuals.
# # Our app offers a cost-effective solution, providing professional guidance at no additional cost beyond the app's initial purchase or subscription.
# # Convenience and Accessibility:
# # Practice yoga anytime, anywhere, without the need for scheduled classes or travel to a studio.
# # Users can tailor their practice to their schedule and preferences, fostering consistency and long-term commitment to yoga.



# import cv2
# import mediapipe as mp
# import numpy as np
# import streamlit as st
# from PIL import Image

# # Initialize Mediapipe Pose
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()


# # Function to detect landmarks in an image
# def detect_landmarks(image):
#     # Convert PIL image to NumPy array
#     image_np = np.array(image)
    
#     # Check if the image has 3 channels (RGB)
#     if image_np.shape[-1] == 3:
#         # Convert RGB to BGR format
#         image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#     else:
#         # Convert grayscale to BGR format
#         image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    
#     # Detect landmarks
#     results = pose.process(image_bgr)
    
#     # Convert image back to RGB
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
#     return results.pose_landmarks, image_rgb


# # Function to compare landmarks between two poses
# def compare_landmarks(landmarks1, landmarks2, threshold=0.2):
#     if landmarks1 is None or landmarks2 is None:
#         return False
    
#     # Extract landmark locations from pose landmarks
#     landmarks1_np = np.array([[lm.x, lm.y] for lm in landmarks1.landmark])
#     landmarks2_np = np.array([[lm.x, lm.y] for lm in landmarks2.landmark])
    
#     # Calculate Euclidean distances between corresponding landmarks
#     distances = np.sqrt(np.sum((landmarks1_np - landmarks2_np) ** 2, axis=1))
    
#     # Compute the average distance
#     avg_distance = np.mean(distances)
    
#     # Compare average distance with threshold
#     return avg_distance < threshold


# # Main Streamlit app
# def main():
#     st.title("Yoga Pose Comparison")

#     # Upload image from computer
#     uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
#         reference_image = Image.open(uploaded_file)

#         # Start webcam capture
#         video = cv2.VideoCapture(0)
#         print("Webcam opened successfully.")

#         # Read reference landmarks
#         reference_landmarks, _ = detect_landmarks(reference_image)
#         if reference_landmarks:
#             print("Reference landmarks detected successfully.")
#         else:
#             print("Failed to detect reference landmarks.")
#             return

#         while video.isOpened():
#             ret, frame = video.read()

#             if not ret:
#                 print("Failed to capture frame from webcam.")
#                 break

#             # Detect landmarks in webcam feed
#             webcam_landmarks, _ = detect_landmarks(frame)
#             if webcam_landmarks:
#                 print("Webcam landmarks detected successfully.")
#             else:
#                 print("Failed to detect webcam landmarks.")
#                 continue

#             # Compare landmarks with reference landmarks
#             if webcam_landmarks:
#                 is_correct_pose = compare_landmarks(webcam_landmarks, reference_landmarks)

#                 # Provide feedback
#                 feedback_text = "Correct pose" if is_correct_pose else "Incorrect pose"
#                 text_color = (0, 255, 0) if is_correct_pose else (0, 0, 255)

#                 # Add text overlay on the frame
#                 cv2.putText(frame, feedback_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

#             # Display frame with feedback
#             cv2.imshow('Yoga Pose Comparison', frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         video.release()
#         cv2.destroyAllWindows()

#     else:
#         st.warning("Please upload an image.")

# if __name__ == "__main__":
#     main()







import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Function to detect landmarks in an image
def detect_landmarks(image):
    # Convert PIL image to NumPy array
    image_np = np.array(image)
    
    # Check if the image has 3 channels (RGB)
    if image_np.shape[-1] == 3:
        # Convert RGB to BGR format
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        # Convert grayscale to BGR format
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    
    # Detect landmarks
    results = pose.process(image_bgr)
    
    # Convert image back to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    return results.pose_landmarks, image_rgb


# Function to compare landmarks between two poses
def compare_landmarks(landmarks1, landmarks2, threshold=0.2):
    if landmarks1 is None or landmarks2 is None:
        return False
    
    # Extract landmark locations from pose landmarks
    landmarks1_np = np.array([[lm.x, lm.y] for lm in landmarks1.landmark])
    landmarks2_np = np.array([[lm.x, lm.y] for lm in landmarks2.landmark])
    
    # Calculate Euclidean distances between corresponding landmarks
    distances = np.sqrt(np.sum((landmarks1_np - landmarks2_np) ** 2, axis=1))
    
    # Compute the average distance
    avg_distance = np.mean(distances)
    
    # Compare average distance with threshold
    return avg_distance < threshold


# Home Page with Pose Comparison
def home():
  
    st.title("Yoga Pose Comparison")

    # Upload image from computer
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        reference_image = Image.open(uploaded_file)

        # Start webcam capture
        video = cv2.VideoCapture(0)
        print("Webcam opened successfully.")

        # Read reference landmarks
        reference_landmarks, _ = detect_landmarks(reference_image)
        if reference_landmarks:
            print("Reference landmarks detected successfully.")
        else:
            print("Failed to detect reference landmarks.")
            return

        while video.isOpened():
            ret, frame = video.read()

            if not ret:
                print("Failed to capture frame from webcam.")
                break

            # Detect landmarks in webcam feed
            webcam_landmarks, _ = detect_landmarks(frame)
            if webcam_landmarks:
                print("Webcam landmarks detected successfully.")
            else:
                print("Failed to detect webcam landmarks.")
                continue

            # Compare landmarks with reference landmarks
            if webcam_landmarks:
                is_correct_pose = compare_landmarks(webcam_landmarks, reference_landmarks)

                # Provide feedback
                feedback_text = "Correct pose" if is_correct_pose else "Incorrect pose"
                text_color = (0, 255, 0) if is_correct_pose else (0, 0, 255)

                # Add text overlay on the frame
                cv2.putText(frame, feedback_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

            # Display frame with feedback
            cv2.imshow('Yoga Pose Comparison', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

    else:
        st.warning("Please upload an image.")

# Yoga in Our Life Page
def yoga_benefits():
    st.title("Yoga in Our Life")
    st.write("Here you can learn about the benefits of different yoga poses.")
    st.write("Yoga is an Indian spiritual and physical practice or discipline whose origins date back to prehistoric times. Contrary to what some may think, Yoga is not just about exercise with the main purpose of improving health and well-being, but it’s also about self-realization")
    st.write("The importance of Yoga in our lives cannot be underestimated. It is a science that focuses on improving not only physical health but also mental and spiritual well-being, which are the foundations of our life.")
    st.write("You will be surprised at how many health benefits Yoga has to offer. From offering relief from stress and weight management to improving your overall health, this ancient practice can help you live an ideal life.")
    st.markdown("[Click here to visit Google](https://www.piesfitnessyoga.com/whats-the-importance-of-yoga-in-our-life/)")


# Comments & Ratings Page
def comments_ratings():
    st.title("Comments & Ratings")
    st.write("Please leave your comments and ratings for our app here.")

    # Text area for the comment
    comment = st.text_area("Write your comment:")

    # Slider for rating (from 1 to 5)
    rating = st.slider("Rate the app (1 - Worst, 5 - Best)", 1, 5)

    # Button to submit the comment and rating
    if st.button("Submit"):
        # Display the submitted comment and rating
        st.write(f"Your comment: {comment}")
        st.write(f"Your rating: {rating}")



# Main function to switch between pages
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "Yoga in Our Life", "Comments & Ratings"])

    if selection == "Home":
        home()
    elif selection == "Yoga in Our Life":
        yoga_benefits()
    elif selection == "Comments & Ratings":
        comments_ratings()

if __name__ == "__main__":
    main()
