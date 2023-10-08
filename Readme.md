Skin.AI Web App

INTRODUCTION
Welcome to the Skin.AI! 
This web application allows users to upload or take a picture of a skin condition, and it provides a diagnosis if a skin condition is detected. This project leverages a computer vision transformer and machine learning techniques to analyze and classify the uploaded images. Additionally, we created an LLM(Large Language Model) called text-bison from Google's Vertex AI to display the diagnosis of the user. The web app was designed using streamlit, and was implemented in Pycharm CE. 

The goal of this project is to provide a user-friendly and accessible tool for individuals who may be concerned about their skin conditions. 

****DISCLAIMER: This web app is for informational purposes only and should not replace professional medical advice***

FEATURES
- Image upload: Users can upload images of their skin conditions from their devices.
- Camera Integration: Users can take pictures using their device's camera to be analyzed.
- Skin Condition Diagnosis: The app uses machine learning to analyze the images and provide a diagnosis if a skin condition is detected.
- User-Friendly Interface: The user interface is designed to be intuitive and easy to use.
- Privacy: We respect user privacy and do not store or share uploaded images.

GETTING STARTED
Before you begin, make sure you have the proper prerequisites installed using:
pip install -r requirements.txt

Additionally, all files should be in HackRU-Fall2023 directory to run

USAGE
Use the following command to run the web app:
streamlit run main.py

HARDSHIPS

- Due to GPU and time limitations, we couldn't train the data as accurate as we would have liked
- streamlit prohibited the use of CSS in Pycharm CE(only in the paid professional version)
- LLM implementation issues
     - First we tried using OpenAi and GPT3, but they required payment, and we would have lost API credits
     - Then we tried open source HuggingFace LLM models(mosaicml/mpt-30b-chat, TheBloke/vicuna-7B-v1.3-GPTQ, microsoft/biogpt, microsoft/BioGPT-Large-PubMedQA, microsoft/BiomedCLIP- 
       PubMedBERT_256-vit_base_patch16_224, microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract) but they were either too complex to implement or deprecated

UPCOMING PLANS

- Resolve training accuracy issues
- Make web app more aesthetically pleasing using CSS in a different IDE
- Build a fully functioning chat bot to give proactive feedback to the user
