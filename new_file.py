import streamlit as st


import matplotlib.pyplot as plt
from PIL import Image
import io
from io import BytesIO
from joblib import load
from new import verifyFace,preprocess_image,findCosineSimiliarity
import matplotlib.pyplot as plt



st.write("Please input first image")
# pic1 = st.file_uploader("Upload your pictures", type=['jpg', 'png'], accept_multiple_files=False)
# bytes_data = pic1.read()
# image = Image.open(io.BytesIO(bytes_data))
# f = plt.figure()
# plt.imshow(bytes_data)
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
# st.pyplot(f)

uploaded_file1 = st.file_uploader("Upload an image", type=['jpg', 'png'], accept_multiple_files=False)
if uploaded_file1 is not None:
    bytes_data = uploaded_file1.read()
    st.write("Filename:", uploaded_file1.name)
    image = Image.open(io.BytesIO(bytes_data))
    f = plt.figure(figsize=(2, 3))
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    st.pyplot(f)
st.write("Please input second image")
#pic2 = st.file_uploader("Upload your pictures", type=['jpg', 'png','jpeg'], accept_multiple_files=False)
uploaded_file2 = st.file_uploader("Upload an image", type=['jpg', 'png','jpeg'], accept_multiple_files=False)
if uploaded_file2 is not None:
    bytes_data = uploaded_file2.read()
    st.write("Filename:", uploaded_file2.name)
    image = Image.open(io.BytesIO(bytes_data))
    f = plt.figure(figsize=(2, 3))
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    st.pyplot(f)
def faceVerification():
    if st.button('Verify Face'):  # Add a button to trigger face verification
            st.write("Face Verification")
            # Ensure 'image' is defined where it's being used or passed as a parameter
            st.write(verifyFace(uploaded_file1, uploaded_file2))  # 'image' still needs to be defined or handled properly
        
            
faceVerification()