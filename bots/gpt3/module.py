import torch
from torch import nn
import openai
# from googletrans import Translator

class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
        openai.api_key =  config.api
        self.chat = False
        if config.gpt3 == 'ada':
          self.engine = 'text-ada-001'
        elif config.gpt3 == 'curie':
          self.engine = 'text-curie-001'
        elif config.gpt3 == 'babbage':
          self.engine = 'text-babbage-001'
        elif config.gpt3 == 'davinci':
          self.engine = 'text-davinci-003'
        elif config.gpt3 == 'turbo':
          # self.model = "gpt-3.5-turbo"
          self.chat = True
        self.interlocutor = config.interlocutor
        # self.translator = Translator()
        self.log_file = 'results/' + config.save_path+'/log.txt'
        # self.log = open('results/' + config.save_path+'/log.txt', 'w+')
        with open(self.log_file, "w+") as f:
          f.write("")
    def make_response(self, prefix_sentences, prompts):
        
        #openai.api_key = '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'
        #openai.api_key =  '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'
        
        #openai.api_key =  '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'
        if self.chat == True:
          log = {}
          log['prompt'] = prompts
          log['prefix'] = prefix_sentences
          with torch.no_grad():
            # messages = []
            reply_string = []
            for i in range(len(prompts)):
              message = []
              message.append({"role": "system", "content": prompts[i]})
              message.append({"role": "user", "content": prefix_sentences[i]})
              # messages.append(message)
              response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=0.0
              )
              reply_string.append(response['choices'][0]['message']['content'])
            for i in range(len(reply_string)):
              reply_string[i] = [reply_string[i].strip()]
            log['AI'] = reply_string
            
            if self.interlocutor:
              pass
            else:
              with open(self.log_file, "a") as f:
                for i in range(len(prompts)):
                  f.write(f'{i}\n{log["prompt"][i]}\n\nHuman: {log["prefix"][i]}\nAI: {log["AI"][i][0]}\n')
                f.write("=========================================\n")
            return reply_string
        else:
          log = {}
          log['prompt'] = prompts
          log['prefix'] = prefix_sentences
          with torch.no_grad():
            sentences = []
            # output_sentences = [tokenizer.encode(x, add_prefix_space=True) for x in output_sentences_string]
            # prompt = [tokenizer.encode(x, add_prefix_space=True) for x in first_input_string]
            
            for i in range(len(prompts)):
                
                #total_string  = "There is office in the following response:" + output_sentences_string[i]
                # total_string  = "Make the following response full of office:" + output_sentences_string[i]
                # total_string = prompts[i] + prefix_sentences[i]
                # sentences.append(f"{total_string}\n\n")
                # s = self.translator.translate(f"{prompts[i]}\n\nHuman: {prefix_sentences[i]}\nAI:").text
                # s_p = self.translator.translate(prompts[i]).text
                # s_s = self.translator.translate(prefix_sentences[i]).text
                # s = s_p + '<|endoftext|>' + s_s

                s = f"{prompts[i]}\n\nHuman:{prefix_sentences[i]}\nAI: "
                # s = f"I need more reply.\n\nHuman:{prefix_sentences[i]}\nAI: "
                sentences.append(s)
                # self.log.write(f'{i}:\n {s}\n')
                # print(f'{i}:\n {s}')
                
            reply_string = []
            # print(sentences)

            response = openai.Completion.create(
                    engine=self.engine,
                    prompt=sentences,
                    temperature=0.0,
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
                # print(f"{i}:\n {reply_string[i]}") 
            log['AI'] = reply_string

            if self.interlocutor:
              sentences = []
              for i in range(len(prompts)):
                  sentences.append(f"Human: {prefix_sentences[i]}\nAI: {reply_string[i][0]}\nHuman: ")
              reply_string = []

              response = openai.Completion.create(
                      engine=self.engine,
                      prompt=sentences,
                      temperature=0.0,
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

              log['interlocutor'] = reply_string
              with open(self.log_file, "a") as f:
                for i in range(len(prompts)):
                  f.write(f'{i}:\n{log["prompt"][i]}\n\nHuman: {log["prefix"][i]}\nAI: {log["AI"][i][0]}\nHuman: {log["interlocutor"][i][0]}\n')
                  # f.write(f'{i}:\n{sentences[i]}{reply_string[i][0]}\n')
                f.write("=========================================\n")
            else:
              with open(self.log_file, "a") as f:
                for i in range(len(prompts)):
                  f.write(f'{i}\n{log["prompt"][i]}\n\nHuman: {log["prefix"][i]}\nAI: {log["AI"][i][0]}\n')
                f.write("=========================================\n")


          return reply_string
