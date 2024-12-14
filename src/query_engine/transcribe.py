##Online using GPT API for demo/quick
from openai import OpenAI
from halo import Halo

class Transcriber:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def transcribe(self, audio_path):
        """Transcribe audio file using OpenAI Whisper"""
        try:
            spinner = Halo(text='Transcribing audio...', spinner='dots')
            spinner.start()
            
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            spinner.succeed('Transcription complete')
            return transcript.text
            
        except Exception as e:
            spinner.fail('Transcription failed')
            raise Exception(f"Transcription error: {str(e)}")

##LOCAL        
# import whisper
# from halo import Halo

# class Transcriber:
#     def __init__(self, model_size="base"):
#         """Initialize the Whisper model locally."""
#         self.model = whisper.load_model(model_size)

#     def transcribe(self, audio_path):
#         """Transcribe audio file using the locally running Whisper model."""
#         try:
#             spinner = Halo(text='Transcribing audio...', spinner='dots')
#             spinner.start()

#             # Load and preprocess the audio
#             audio = whisper.load_audio(audio_path)
#             audio = whisper.pad_or_trim(audio)

#             # Generate log-Mel spectrogram
#             mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

#             # Decode the audio
#             options = whisper.DecodingOptions(fp16=False)
#             result = whisper.decode(self.model, mel, options)

#             spinner.succeed('Transcription complete')
#             return result.text

#         except Exception as e:
#             spinner.fail('Transcription failed')
#             raise Exception(f"Transcription error: {str(e)}")
