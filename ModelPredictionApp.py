import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
from keras.models import load_model
import tensorflow as tf
import imageio
import base64
import pandas as pd
import matplotlib.pyplot as plt

#Main function
def main(): 
    imagePrediction(magnificationFactorValue)
#def showModel():
    
    
#Image prediction by Model    
def imagePrediction(magnificationFactorValue): #frameA_Image, frameB_Image, magnificationFactorValue):
    frameA_Image = []
    frameB_Image = []   
    #trainedModel = tf.saved_model.load_model('saved model 20k epoch10')    
    frameA_Image = st.file_uploader("Please Upload a Input Image below for frame A...", type=["jpg", "png", "mp4", "avi"])
    frameB_Image = st.file_uploader("Please Upload a Input Image below for frame B...", type=["jpg", "png", "mp4", "avi"])
    #imagePrediction(frameA_Image, frameB_Image, magnificationFactorValue)
    predict_button = st.button("Image Predicted by Model")
    img_A = ''
    img_predicted= ''
    col1, col2, col3 = st.columns([6, 4, 6])
    trainedModel = load_model('saved model 20k epoch10')
    if predict_button:
        if frameA_Image is not None:
            if frameB_Image is not None:
                frameA_Image_Array = np.array(Image.open(frameA_Image))
                frameB_Image_Array = np.array(Image.open(frameB_Image))
                #frameA_Image_Array = cv2.imread(os.path.join("tempDir",img_name_A))
                #frameB_Image_Array = cv2.imread(os.path.join("tempDir",img_name_B))
                magnification_factor_value = tf.Variable([[[magnificationFactorValue]]])
                magnification_factor_value = tf.expand_dims(magnification_factor_value,axis=0)
                im1= tf.expand_dims(frameA_Image_Array,axis=0)
                im2= tf.expand_dims(frameB_Image_Array,axis=0)
                modelPredictImage = trainedModel.predict(
                    [im1, im2, magnification_factor_value[:1]])               
                image_norm = cv2.normalize(np.squeeze(modelPredictImage[0]), None, alpha=0,beta=1, norm_type=cv2.NORM_MINMAX)
                col1.caption('Predicted Image')
                col1.image([np.squeeze(image_norm)], use_column_width=False,clamp=True, channels='RGB')
                predictedImageByModel = np.squeeze(image_norm)
                SaveImages(frameA_Image, modelPredictImage) #[np.squeeze(image_norm)])

def showResult():
    col1, col2, col3 = st.columns([6, 4, 6])
    col1.caption('Predicted Image')
    col1.image(os.path.join("ModelPredictImages","predictedImage.PNG"), use_column_width=False,clamp=True, channels='RGB')
    col3.caption('gif Image')
    file_ = open(os.path.join("ModelPredictImages","finalGifByModel.gif"), "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    col3.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
                unsafe_allow_html=True)              
 
    
#Creating gif using Image list
def CreategifFromImages(pathOfImage1,pathOfImage2,gif_name):
    images_frameAB = []
    pathOfImage1 = os.path.join("ModelPredictImages",pathOfImage1)
    pathOfImage2 = os.path.join("ModelPredictImages",pathOfImage2)
    images_frameAB.append(imageio.imread(pathOfImage1))
    images_frameAB.append(imageio.imread(pathOfImage2))
    imageio.mimsave(os.path.join("ModelPredictImages",gif_name), images_frameAB,'GIF',duration=0.5)


#Save Input and Predicted Image for creating gif file
def SaveImages(inputImage, predictedImage):
    if inputImage is not None:
        if predictedImage is not None:
            img_A = "frameAImage.PNG"
            img_Predicted = "predictedImage.PNG"
            inputFileDetails = {"FileName":img_A,"FileType":"PNG"}
            predictedFileDetails = {"FileName":predictedImage,"FileType":"PNG"}
            with open(os.path.join("ModelPredictImages",img_A),"wb") as f: 
                 f.write(inputImage.getbuffer())   
            #imagePredicted = Image.fromarray(predictedImage,'BGR')    
            #imagePredicted.save(os.path.join("ModelPredictImages",img_Predicted))
            im = Image.fromarray(np.squeeze(predictedImage.astype(np.uint8)),'RGB')
            im.save(os.path.join("ModelPredictImages",img_Predicted))            
            CreategifFromImages(img_A,img_Predicted,"finalGifByModel.gif")
            st.success("Saved gif file")

#loss Curve Plot
def plotLossCurve(df):
    plt.figure()
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(df["loss"],label='Training Loss')
    plt.plot(df["val_loss"],label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    st.pyplot(plt)

    
#Accuracy Curve Plot
def plotAccuracyCurve(df):
    plt.figure()
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(df["accuracy"],label='Accuracy')
    plt.plot(df["val_accuracy"],label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    st.pyplot(plt)

#SideBar on UI
sideb = st.sidebar
sideb.markdown("Please click here")
magnificationFactorValue = sideb.slider('Please select Magnification Factor from slider', min_value=1, max_value=100)
lossCurveflag = sideb.button("Loss Curve")
accuracyCurveflag = sideb.button("Accuracy Curve") 
if "button_clicked" not in st.session_state:    
    st.session_state.button_clicked = False
predictImage = sideb.button("Predict Image")
showPredictedImage = sideb.button("Show Image")
st.title("Detect Microscopic Motion in Mechanical Equipment using Deep-Learning and smartphones")
model_loading_state = st.text('Getting things ready! Please wait...')
model_loading_state.text('The AI is all saved model 20k epoch10 set!')
if __name__ == '__main__':
     main()        
    
df = pd.read_csv('log.csv')
if lossCurveflag:
    plotLossCurve(df)
if accuracyCurveflag:
    plotAccuracyCurve(df)
#if predictImage:
    #imagePrediction(magnificationFactorValue)
if showPredictedImage:
    showResult()
#st.markdown(hide_menu_style, unsafe_allow_html=True)
#if predictImage:
    #if uploadedImageList is not None:
        #imagePrediction(predictImage,magnificationFactorValue)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
