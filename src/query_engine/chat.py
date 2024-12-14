from openai import OpenAI
import textwrap

class ChatBot:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.messages = [
            {"role": "system", "content": """You are an AI assistant specialized in processing and analyzing transcribed audio content. Your primary functions are:

                                              1. Process incoming text chunks(can be more than one) from transcribed audio (lectures, recordings, etc.)
                                              2. Build and maintain a comprehensive understanding of the entire content
                                              3. Store key information, concepts, and relationships from each chunk
                                              4. Maintain context continuity across all processed chunks
                                              5. Answer queries by synthesizing information from all processed chunks

                                              For each new chunk you receive:
                                              - Analyze the content thoroughly
                                              - Link concepts with previously received chunks
                                              - Identify main topics and subtopics
                                              - Note any specific examples, definitions, or important details
                                              - Preserve chronological flow of information

                                              When answering queries:
                                              - Draw from the complete accumulated context
                                              - Provide accurate, relevant responses based solely on the processed chunks
                                              - Maintain consistency with all provided information
                                              - Reference specific details from appropriate chunks when necessary
                                              """}
        ]

    def simple_chunk(self, text, size=4000):
        text = ' '.join(text.split())
        return textwrap.wrap(text, width=size)
    
    def set_context(self, context):
        """Process and set the context for future questions"""
        try:
            chunks = self.simple_chunk(context)
            
            for i, chunk in enumerate(chunks):
                self.messages.append({
                    "role": "user",
                    "content": f"New information: {chunk}"
                })
                self.messages.append({
                    "role": "assistant",
                    "content": "I have processed and understood this new information."
                })
                
        except Exception as e:
            raise Exception(f"Context setting error: {str(e)}")
    
    def ask(self, question):
        """Ask question based on the previously set context"""
        try:
            # Add the question to existing conversation
            self.messages.append({
                "role": "user",
                "content": f"Based on all previous information, please answer: {question}"
            })

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                max_tokens=200,
                temperature=0.7
            )

            answer = response.choices[0].message.content
            
            self.messages.append({
                "role": "assistant",
                "content": answer
            })
            
            return answer
            
        except Exception as e:
            return "I apologize, but I couldn't process your question. Could you please try asking it again?"

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import textwrap

# class ChatBot:
#     def __init__(self, model_name="gpt2"):
#         """Initialize the chatbot with a local model."""
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
        
#         # Store the conversation context
#         self.context = []

#         # Initial system prompt
#         self.system_prompt = (
#             "You are an AI assistant specialized in processing and analyzing transcribed audio content. Your primary functions are:\n\n"
#             "1. Process incoming text chunks from transcribed audio (lectures, recordings, etc.)\n"
#             "2. Build and maintain a comprehensive understanding of the entire content\n"
#             "3. Store key information, concepts, and relationships from each chunk\n"
#             "4. Maintain context continuity across all processed chunks\n"
#             "5. Answer queries by synthesizing information from all processed chunks\n\n"
#             "For each new chunk you receive:\n"
#             "- Analyze the content thoroughly\n"
#             "- Link concepts with previously received chunks\n"
#             "- Identify main topics and subtopics\n"
#             "- Note any specific examples, definitions, or important details\n"
#             "- Preserve chronological flow of information\n\n"
#             "When answering queries:\n"
#             "- Draw from the complete accumulated context\n"
#             "- Provide accurate, relevant responses based solely on the processed chunks\n"
#             "- Maintain consistency with all provided information\n"
#             "- Reference specific details from appropriate chunks when necessary."
#         )

#         self.context.append(self.system_prompt)

#     def simple_chunk(self, text, size=4000):
#         """Split the text into manageable chunks."""
#         text = ' '.join(text.split())
#         return textwrap.wrap(text, width=size)

#     def set_context(self, context):
#         """Process and set the context for future questions."""
#         try:
#             chunks = self.simple_chunk(context)

#             for chunk in chunks:
#                 self.context.append(f"New information: {chunk}")
#                 self.context.append("I have processed and understood this new information.")

#         except Exception as e:
#             raise Exception(f"Context setting error: {str(e)}")

#     def ask(self, question):
#         """Ask a question based on the previously set context."""
#         try:
#             # Combine the context and the question
#             input_text = "\n".join(self.context + [f"Question: {question}", "Answer:"])

#             # Tokenize and generate the response
#             inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
#             outputs = self.model.generate(
#                 inputs.input_ids, 
#                 max_length=1024, 
#                 temperature=0.7, 
#                 pad_token_id=self.tokenizer.eos_token_id
#             )

#             # Decode the generated text
#             answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

#             # Extract the answer portion
#             answer = answer.split("Answer:")[-1].strip()

#             # Update context
#             self.context.append(f"Question: {question}")
#             self.context.append(f"Answer: {answer}")

#             return answer

#         except Exception as e:
#             return "I apologize, but I couldn't process your question. Could you please try asking it again?"
