from google.colab import drive
drive.mount('/content/drive')

# Create project folder
!mkdir -p /content/drive/MyDrive/MedicalAI_Project/data

!pip install kaggle

from google.colab import files
files.upload()  # Upload kaggle.json

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download chest X-ray pneumonia dataset
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p /content/drive/MyDrive/MedicalAI_Project/data --unzip

import os

data_dir = "/content/drive/MyDrive/MedicalAI_Project/data/chest_xray"

for split in ['train', 'val', 'test']:
    for label in ['NORMAL', 'PNEUMONIA']:
        folder = os.path.join(data_dir, split, label)
        count = len(os.listdir(folder))
        print(f"{split}/{label}: {count} images")

import matplotlib.pyplot as plt
import cv2

sample_path = os.path.join(data_dir, 'train', 'PNEUMONIA')
sample_images = os.listdir(sample_path)[:6]

plt.figure(figsize=(12,6))
for i, img_file in enumerate(sample_images):
    img = cv2.imread(os.path.join(sample_path, img_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title("PNEUMONIA")
    plt.axis('off')
plt.tight_layout()
plt.show()

#Preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image size for resizing
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Data generators
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    os.path.join(data_dir, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

#Built CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#train the model
EPOCHS = 20

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

model_path = "/content/drive/MyDrive/MedicalAI_Project/pneumonia_cnn_model.h5"
model.save(model_path)
print("Model saved to Drive")

#Accuracy and loss graphs
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#Train GAN to Generate Normal Chest X-Rays
#Collect Only NORMAL Images

import glob
import cv2
import numpy as np

normal_dir = os.path.join(data_dir, 'train', 'NORMAL')
normal_images = []

for path in glob.glob(normal_dir + "/*.jpeg"):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))  # Small size for GAN
    normal_images.append(img)

normal_images = np.array(normal_images)
normal_images = (normal_images - 127.5) / 127.5  # Normalize to [-1, 1]
normal_images = np.expand_dims(normal_images, axis=-1)  # Add channel

print("Loaded NORMAL images for GAN:", normal_images.shape)

import tensorflow as tf
from tensorflow.keras import layers

# Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', activation='tanh')
    ])
    return model

# Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=(64,64,1)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

BATCH_SIZE = 64
noise_dim = 100
EPOCHS = 2000  # Lower for quick testing

# Batch dataset
BUFFER_SIZE = normal_images.shape[0]
train_dataset = tf.data.Dataset.from_tensor_slices(normal_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = build_generator()
discriminator = build_discriminator()

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

import matplotlib.pyplot as plt
import os

# Output folder
output_dir = "/content/drive/MyDrive/MedicalAI_Project/generated_normal_images"
os.makedirs(output_dir, exist_ok=True)

seed = tf.random.normal([16, noise_dim])

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        img = (predictions[i, :, :, 0] + 1) / 2  # from [-1,1] to [0,1]
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    file_path = os.path.join(output_dir, f'image_epoch_{epoch}.png')
    plt.savefig(file_path)
    plt.close()

# Training loop
for epoch in range(1, EPOCHS+1):
    for image_batch in train_dataset:
        g_loss, d_loss = train_step(image_batch)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Gen Loss: {g_loss:.4f}, Disc Loss: {d_loss:.4f}")
        generate_and_save_images(generator, epoch, seed)

#Generate Medical Reports Using T5
#Install & Import Required Libraries
!pip install -q transformers

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

save_path = "/content/drive/MyDrive/MedicalAI_Project/T5"

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print("T5 model and tokenizer saved to Drive.")

from transformers import T5Tokenizer, T5ForConditionalGeneration

load_path = "/content/drive/MyDrive/MedicalAI_Project/T5"

tokenizer = T5Tokenizer.from_pretrained(load_path)
model = T5ForConditionalGeneration.from_pretrained(load_path)
print("T5 model and tokenizer loaded from Drive.")

def generate_t5_report(prediction, confidence):
    clinical_input = f"""
    Chest X-ray shows signs of {prediction.lower()} with {confidence*100:.1f}% confidence.
    Patient may require further clinical correlation and intervention.
    """
    input_text = "summarize: " + clinical_input.strip()

    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(input_ids, max_length=60, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Add structure
    final_report = f"""
     Radiology Report

Findings: {summary}

Impression: Suggestive of {prediction.upper()}
Recommendation: Correlate clinically. Consider further investigation if symptoms persist.
""".strip()

    return final_report

#Test the function
prediction = "Pneumonia"
confidence = 0.91

report = generate_t5_report(prediction, confidence)
print("Generated Report:\n", report)

#Gradio
!pip install -q gradio
import gradio as gr
from tensorflow.keras.models import load_model

# Load CNN model from Drive
cnn_model = load_model("/content/drive/MyDrive/MedicalAI_Project/pneumonia_cnn_model.h5")

from transformers import T5Tokenizer, T5ForConditionalGeneration

t5_path = "/content/drive/MyDrive/MedicalAI_Project/T5"
tokenizer = T5Tokenizer.from_pretrained(t5_path)
model = T5ForConditionalGeneration.from_pretrained(t5_path)

import numpy as np
from PIL import Image

def preprocess_image(image):
    image = image.resize((150,150)).convert('RGB')
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def generate_t5_report(prediction, confidence):
    if prediction.lower() == "normal":
        clinical_input = f"""
        Chest X-ray appears normal with {confidence*100:.2f}% confidence.
        No visible abnormalities are noted.
        """
    else:
        clinical_input = f"""
        Chest X-ray shows signs of {prediction.lower()} with {confidence*100:.2f}% confidence.
        Patient may require further clinical evaluation.
        """

    input_text = "summarize: " + clinical_input.strip()

    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(input_ids, max_length=60, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    final_report = f"""
Radiology Report

Findings: {summary}

Impression: Suggestive of {prediction.upper()}
Confidence: {confidence*100:.2f}%

Recommendation: {"No further action required at this time." if prediction.lower() == "normal" else "Correlate clinically. Consider follow-up imaging or additional tests as needed."}
""".strip()

    return final_report


def predict_and_report(image):
    preprocessed = preprocess_image(image)
    prediction = cnn_model.predict(preprocessed)[0][0]

    label = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    report = generate_t5_report(label, confidence)

    return label, f"{confidence*100:.2f}%", report

interface = gr.Interface(
    fn=predict_and_report,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Doctor-style Medical Report")
    ],
    title="AI-Powered Pneumonia Detection & Report Generator",
    description="Upload a chest X-ray to detect pneumonia and receive an automated report using a CNN + T5 model.",
)
interface.launch(share=True)
