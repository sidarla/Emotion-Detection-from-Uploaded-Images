import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Set page layout
st.set_page_config(page_title="Teachable Machine", layout="wide")

# Load the trained model architecture
class CNN_emotion(nn.Module):
    def __init__(self):
        super(CNN_emotion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 17 * 17, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 100)
        self.fc5 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, 7)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        X = self.conv1(X)
        X = self.relu(self.maxpool1(X))
        X = self.relu(self.conv2(X))
        X = self.relu(self.conv3(X))
        X = self.relu(self.conv4(X))
        X = X.view(X.size(0), -1)
        X = self.relu(self.fc1(X))
        X = self.dropout(self.relu(self.fc2(X)))
        X = self.dropout(self.relu(self.fc3(X)))
        X = self.relu(self.fc4(X))
        X = self.relu(self.fc5(X))
        X = self.fc6(X)
        return X

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = CNN_emotion()
    model.load_state_dict(torch.load('emotion_detection_model.pth', map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model

model = load_model()

# Define emotion categories
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define transformation for input image
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Grayscale normalization
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Streamlit app



#Background-color
st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #dec3ab;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

#tabs section
tab1, tab2= st.tabs(["Home", "Upload Image"])

with tab1:
    

    # Styling for custom fonts
    st.markdown(
        """
        <style>
        .title {
            font-family: 'Arial', sans-serif;
            font-size: 3rem;
            font-weight: 700;
            color: #1a73e8; /* Blue */
        }
        .subtitle {
            font-family: 'Arial', sans-serif;
            font-size: 1.5rem;
            font-weight: 500;
            marin-bottom:3px
            color: #333333;
        }
        .description {
            font-family: 'Arial', sans-serif;
            font-size: 1.1rem;
            font-weight: 400;
            color: #666666;
            margin-bottom:8px
        }
        .button {
            background-color: #1a73e8;
            color: white;
            padding: 10px 20px;
            font-size: 1.2rem;
            font-weight: 600;
            text-align: center;
            border-radius: 8px;
            display: inline-block;
        }
        .button:hover {
            background-color: #0c5fc1;
            cursor: pointer;
        }
        .footer-icons {
            font-size: 1.2rem;
            margin-top: 20px;
            display: flex;
            gap: 15px;
        }
        .footer-icons img {
            height: 35px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Main content area
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown('<div class="title">Emotion Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Train a computer to recognize your own images, Videos</div>', unsafe_allow_html=True)
        st.markdown('<div class="description">A fast, easy way to create CNN models for your sites, apps, and more – no expertise or coding required.</div>', unsafe_allow_html=True)
        
        # Get Started button
        st.markdown('<div class="button">Get Started</div>', unsafe_allow_html=True)

        
    with col2:

        # Load and display the image with border-radius
        image = Image.open("emotion_home_images.png")
        st.image(image, caption="Emotion Home", use_column_width=True, clamp=True)


with tab2:

    st.markdown(
                """
                <style>
                .custom-title {
                    color: #0c0d0d;  
                    font-size: 38px;
                    font-family: 'playfair display';
                    text-align:center;
                }
                .result_predict {
                    color: #c20c2a;  
                    font-size: 35px;
                    font-family: 'bree serif';
                    text-align:center;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

    st.markdown(f'<h1 class="custom-title">Emotion Detection from Uploaded Images</h1>', unsafe_allow_html=True)
    st.markdown('<div class="description">Upload an image to detect the emotion of the person in the image.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image and run prediction
        image_tensor = transform_image(image)
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.argmax(output, dim=1)
            predicted_emotion = EMOTIONS[prediction.item()]

        #st.write(f"Predicted Emotion: {predicted_emotion}")
        st.markdown(f'<h2 class="result_predict">Predicted Emotion: {predicted_emotion}</h2>', unsafe_allow_html=True)
