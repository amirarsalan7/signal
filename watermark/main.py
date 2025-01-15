import numpy as np
import pywt
import wave
import cv2

# Read the image
image_path = 'IMG_2.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale

# Read the audio
audio_path = 'mixkit-classic-alarm-995.wav'  # Replace with your audio path
with wave.open(audio_path, 'rb') as audio_file:
    params = audio_file.getparams()
    audio_frames = audio_file.readframes(params.nframes)
    audio_signal = np.frombuffer(audio_frames, dtype=np.int16)

# Normalize the audio signal
audio_signal = (audio_signal - np.min(audio_signal)) / (np.max(audio_signal) - np.min(audio_signal))

# Step 1: Perform DWT on the image
coeffs_image = pywt.dwt2(image, 'haar')
LL, (LH, HL, HH) = coeffs_image

# Step 2: Divide the audio signal into chunks
chunk_size = HH.size
audio_chunks = [audio_signal[i:i + chunk_size] for i in range(0, len(audio_signal), chunk_size)]

# Ensure the last chunk is properly sized
if len(audio_chunks[-1]) < chunk_size:
    audio_chunks[-1] = np.pad(audio_chunks[-1], (0, chunk_size - len(audio_chunks[-1])), mode='constant')

# Step 3: Embed each audio chunk into the HH sub-band
alpha = 0.01  # Embedding strength
HH_watermarked = HH.copy()

for i, chunk in enumerate(audio_chunks):
    reshaped_chunk = chunk.reshape(HH.shape)
    HH_watermarked += alpha * reshaped_chunk

# Step 4: Reconstruct the watermarked image
coeffs_watermarked = (LL, (LH, HL, HH_watermarked))
watermarked_image = pywt.idwt2(coeffs_watermarked, 'haar')

# Clip and convert to uint8 for saving
watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
cv2.imwrite('watermarked_image.png', watermarked_image)

# Step 5: Extract the audio signal from the watermarked image
coeffs_extracted = pywt.dwt2(watermarked_image, 'haar')
_, (_, _, HH_extracted) = coeffs_extracted

# Extract each audio chunk from the HH sub-band
audio_chunks_extracted = []

for i, chunk in enumerate(audio_chunks):
    extracted_chunk = (HH_extracted - HH) / alpha
    audio_chunks_extracted.append(extracted_chunk.flatten())

# Combine all extracted chunks
extracted_audio = np.concatenate(audio_chunks_extracted)[:len(audio_signal)]

# Normalize the extracted audio
extracted_audio = (extracted_audio - np.min(extracted_audio)) / (np.max(extracted_audio) - np.min(extracted_audio))
extracted_audio = (extracted_audio * 32767).astype(np.int16)

# Save the extracted audio
with wave.open('extracted_audio.wav', 'wb') as output_audio:
    output_audio.setparams(params)
    output_audio.writeframes(extracted_audio.tobytes())

print("Watermarking and extraction complete.")
