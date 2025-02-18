import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Load ONNX model
onnx_model_path = "face_recognition.onnx"
session = ort.InferenceSession(onnx_model_path)

# Load known face embeddings
known_embeddings = np.load("known_embeddings.npy")
known_names = np.load("known_names.npy")

# Ensure known_embeddings is not empty
if known_embeddings.size == 0:
    raise ValueError("The known_embeddings.npy file is empty! Ensure it contains valid face embeddings.")

print("Known embeddings shape:", known_embeddings.shape)


# Function to preprocess the image
def preprocess_image(image):
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided. Ensure the image is loaded correctly.")

    image = cv2.resize(image, (299, 299))  # Resize to match model input size
    image = image.astype('float32') / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    print("Preprocessed image shape:", image.shape)  # Debugging
    return image

# Function to extract face embedding
def get_embedding(image):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    processed_image = preprocess_image(image)
    embedding = session.run([output_name], {input_name: processed_image})[0]
    return embedding


def classify_face(embedding):
    # Ensure embedding is 2D
    embedding = embedding.reshape(1, -1)
    
    # Compute cosine similarity
    similarities = cosine_similarity(embedding, known_embeddings)[0]
    
    # Get best match
    best_match_idx = np.argmax(similarities)
    best_match_name = known_names[best_match_idx]
    best_match_score = similarities[best_match_idx]

    return best_match_name, best_match_score


# Streamlit UI
st.title("Face Recognition System")

# Sidebar options
option = st.sidebar.selectbox("Choose an option", ["Upload Image", "Live Camera"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Extract embedding and classify
        embedding = get_embedding(image)
        label, confidence = classify_face(embedding)
        
        st.write(f"Recognition Result: {label}")
        st.write(f"Confidence: {confidence:.2f}")

elif option == "Live Camera":
    st.write("Live Camera Face Recognition")
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])  # Placeholder for displaying the video stream

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video.")
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract embedding and classify
            embedding = get_embedding(frame_rgb)
            label, confidence = classify_face(embedding)
            
            # Display results
            cv2.putText(frame_rgb, f"{label} ({confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            FRAME_WINDOW.image(frame_rgb)
        
        cap.release()

# To Run The App ==> python -m streamlit run app.py


# import streamlit as st
# import cv2
# import numpy as np
# import onnxruntime as ort
# from PIL import Image

# # Load ONNX model
# onnx_model_path = "face_recognition.onnx"
# session = ort.InferenceSession(onnx_model_path)

# # Function to preprocess the image
# def preprocess_image(image):
#     image = cv2.resize(image, (299, 299))  # Resize to match model input size
#     image = image.astype('float32') / 255.0  # Normalize pixel values
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# # Function to perform face recognition
# def recognize_face(image):
#     input_name = session.get_inputs()[0].name
#     output_name = session.get_outputs()[0].name
#     processed_image = preprocess_image(image)
#     results = session.run([output_name], {input_name: processed_image})
#     return results[0]

# # Streamlit UI
# st.title("Face Recognition System")

# # Sidebar options
# option = st.sidebar.selectbox("Choose an option", ["Upload Image", "Live Camera"])

# if option == "Upload Image":
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
#     if uploaded_file is not None:
#         image = np.array(Image.open(uploaded_file))
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         # Perform face recognition
#         results = recognize_face(image)
#         st.write("Recognition Results:")
#         st.write(results)  # Display raw results (you can format this better)

# elif option == "Live Camera":
#     st.write("Live Camera Face Recognition")
#     run = st.checkbox("Start Camera")
#     FRAME_WINDOW = st.image([])  # Placeholder for displaying the video stream

#     if run:
#         cap = cv2.VideoCapture(0)
#         while run:
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("Failed to capture video.")
#                 break

#             # Perform face recognition on the frame
#             results = recognize_face(frame)

#             # Display the frame with recognition results
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
#             FRAME_WINDOW.image(frame)

#         cap.release()


# # To Run The App ==> python -m streamlit run app.py 
# # pip install opencv-python
# # pip install onnxruntime --upgrade
