import os
import cmd
import argparse
from dotenv import load_dotenv 
load_dotenv(".env")

from src.denoiser.processor import AudioProcessor
from src.query_engine.transcribe import Transcriber
from src.query_engine.chat import ChatBot

class AudioCLI(cmd.Cmd):
    intro = 'Welcome to Audio Assistant. Type help or ? to list commands.\n'
    prompt = '(audio) '
    
    def __init__(self, model_path='models/best_model.keras', openai_key=None):
        super().__init__()
        print("\nInitializing Audio Assistant...")
        self.processor = AudioProcessor(model_path)
        self.transcriber = Transcriber(openai_key)
        self.chatbot = ChatBot(openai_key)
        self.current_transcript = None
        print("\nAudio Assistant ready for queries!")
    
    def do_denoise(self, arg):
        'Denoise an audio file: denoise input_path [output_path]'
        args = arg.split()
        if not args:
            print("Please provide input file path")
            return
        
        try:
            input_path = args[0]
            output_path = args[1] if len(args) > 1 else os.environ.get('DEFAULT_OUTPUT_PATH')
            cleaned_file = self.processor.denoise_audio(input_path, output_path)
            print(f"Cleaned audio saved as: {cleaned_file}")
        except Exception as e:
            print(f"Error processing file: {e}")
    
    def do_transcribe(self, arg):
        'Transcribe an audio file: transcribe audio_path'
        if not arg:
            if not os.path.exists(os.environ.get('DEFAULT_OUTPUT_PATH')):
                return
            arg = os.environ.get('DEFAULT_OUTPUT_PATH')

        try:
            self.current_transcript = self.transcriber.transcribe(arg)
        except Exception as e:
            print(f"Error transcribing file: {e}")
    
    def do_ask(self, arg):
        'Ask a question about the transcribed audio: ask "your question here"'
        if not self.current_transcript:
            print("Please transcribe an audio file first using the transcribe command")
            return
        
        if not arg:
            print("Please provide a question")
            return
        
        try:
            # Set context if not already set
            if not hasattr(self.chatbot, 'messages') or len(self.chatbot.messages) <= 1:
                self.chatbot.set_context(self.current_transcript)
            
            # Get response
            response = self.chatbot.ask(arg)
            print("\nAnswer:", response)
            
        except Exception as e:
            print(f"Error getting response: {e}")

    def do_exit(self, arg):
        'Exit the application'
        print("Goodbye!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Audio Assistant')
    parser.add_argument('--model', default='models/best_model.keras', 
                       help='Path to trained model')

    args = parser.parse_args()
    AudioCLI(model_path=args.model, openai_key=os.environ.get('OPENAI_KEY')).cmdloop()

if __name__ == '__main__':
    main()