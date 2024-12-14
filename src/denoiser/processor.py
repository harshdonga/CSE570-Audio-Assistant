import os
import time
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from halo import Halo
import noisereduce as nr
from keras.models import load_model

from .utils import extract_features, NOISE_MAPPING, LABEL_DICT

NOISE_BASE_PATH = os.path.join(os.getcwd(), "src/denoiser/noise")

class AudioProcessor:
    def __init__(self, model_path='models/best_model.keras'):
        self.model = load_model(model_path)
    
    def _get_top_predictions(self, predictions, top_n=3):
        top_indices = predictions.argsort()[-top_n:][::-1]
        return [list(NOISE_MAPPING.keys())[i] for i in top_indices]
    
    def predict_noise_type(self, audio_path):
        spinner = Halo(text='Analyzing audio...', spinner='dots')
        spinner.start()

        features = extract_features(audio_path)
        x_test = np.array([features])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        predictions = self.model.predict(x_test)
        spinner.succeed('Analyzied noise profiles')
        return self._get_top_predictions(predictions[0])

    def denoise_audio(self, audio_path, output_path='clean.wav', target_db=-3):
        noise_types = self.predict_noise_type(audio_path)
        data, sr = librosa.load(audio_path)
        
        # Denoising process
        with tqdm(total=len(noise_types), desc="Denoising audio ", unit="noise") as pbar:
            for noise_type in noise_types:
                noise_files = NOISE_MAPPING[noise_type]
                for noise_file in noise_files:
                    noise, sr2 = librosa.load(os.path.join(NOISE_BASE_PATH, noise_file))
                    data = nr.reduce_noise(y=data, y_noise=noise, sr=sr2)
                pbar.update(1)
                pbar.set_postfix({"Current Noise": LABEL_DICT[noise_type]})

        
        # Volume normalization with progress
        with tqdm(total=3, desc="Normalizing volume", unit="step") as pbar:
            pbar.set_postfix({"Status": "Normalizing range"})
            data = librosa.util.normalize(data)
            pbar.update(1)
            
            pbar.set_postfix({"Status": "Adjusting peak amplitude"})
            peak = np.abs(data).max()
            target_peak = librosa.db_to_amplitude(target_db)
            scaling_factor = target_peak / peak
            data = data * scaling_factor
            pbar.update(1)
            
            pbar.set_postfix({"Status": "Saving audio"})
            data = np.clip(data, -1.0, 1.0)
            sf.write(output_path, data, sr)
            pbar.update(1)

        return output_path