import torch
from torch import nn
import openai

class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
        openai.api_key =  config.api
        if config.gpt3 == 'ada':
          self.engine = 'text-ada-001'
        elif config.gpt3 == 'davinci':
          self.engine = 'text-davinci-003'
        self.interlocutor = config.interlocutor

    def make_response(self, prefix_sentences, prompts):
        
        #openai.api_key = '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'
        #openai.api_key =  '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'
        
        #openai.api_key =  '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'
        with torch.no_grad():
          sentences = []
          # output_sentences = [tokenizer.encode(x, add_prefix_space=True) for x in output_sentences_string]
          # prompt = [tokenizer.encode(x, add_prefix_space=True) for x in first_input_string]
          for i in range(len(prompts)):
              
              #total_string  = "There is office in the following response:" + output_sentences_string[i]
              # total_string  = "Make the following response full of office:" + output_sentences_string[i]
              # total_string = prompts[i] + prefix_sentences[i]
              # sentences.append(f"{total_string}\n\n")
              sentences.append(f"{prompts[i]}\n\nHuman: {prefix_sentences[i]}\nAI:")
          reply_string = []
          print(sentences)

          response = openai.Completion.create(
                  engine=self.engine,
                  prompt=sentences,
                  temperature=0.9,
                  max_tokens=150,
                  top_p=1,
                  frequency_penalty=0,
                  presence_penalty=0.6,
                  stop=[" Human:", " AI:"]
                  )
          for i in range(len(sentences)):
              reply_string.append(response['choices'][i]['text'])
          
          for i in range(len(reply_string)):
              reply_string[i] = [reply_string[i].strip()]
          
          if self.interlocutor:
            sentences = []
            for i in range(len(prompts)):
                sentences.append(f"Human: {prefix_sentences[i]}\nAI: {reply_string[0][i]}")
            reply_string = []

            response = openai.Completion.create(
                    engine=self.engine,
                    prompt=sentences,
                    temperature=0.9,
                    max_tokens=150,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0.6,
                    stop=[" Human:", " AI:"]
                    )
            for i in range(len(sentences)):
                reply_string.append(response['choices'][i]['text'])

            for i in range(len(reply_string)):
                reply_string[i] = [reply_string[i].strip()]

        return reply_string
