import os
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from os.path import join
import re
from argparse import ArgumentParser
import wandb
import openai

from transformers import (
  BertTokenizerFast,
  AutoModel,
  GPT2LMHeadModel,
)

import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
# from chat_load import post_set
from optim import Adam
from dataset import ChineseDataset

import tensorflow as tf

import string
from tqdm import tqdm
torch.manual_seed(100)

bad_word = ["4r5e", "5h1t", "5hit", "a55", "anal", "anus", "ar5e", "arrse", "arse", "ass", "ass-fucker", "asses", "assfucker", "assfukka", "asshole", "assholes", "asswhole", "a_s_s", "b!tch", "b00bs", "b17ch", "b1tch", "ballbag", "balls", "ballsack", "bastard", "beastial", "beastiality", "bellend", "bestial", "bestiality", "bi+ch", "biatch", "bitch", "bitcher", "bitchers", "bitches", "bitchin", "bitching", "bloody", "blow job", "blowjob", "blowjobs", "boiolas", "bollock", "bollok", "boner", "boob", "boobs", "booobs", "boooobs", "booooobs", "booooooobs", "breasts", "buceta", "bugger", "bum", "bunny fucker", "butt", "butthole", "buttmuch", "buttplug", "c0ck", "c0cksucker", "carpet muncher", "cawk", "chink", "cipa", "cl1t", "clit", "clitoris", "clits", "cnut", "cock", "cock-sucker", "cockface", "cockhead", "cockmunch", "cockmuncher", "cocks", "cocksuck", "cocksucked", "cocksucker", "cocksucking", "cocksucks", "cocksuka", "cocksukka", "cok", "cokmuncher", "coksucka", "coon", "cox", "crap", "cum", "cummer", "cumming", "cums", "cumshot", "cunilingus", "cunillingus", "cunnilingus", "cunt", "cuntlick", "cuntlicker", "cuntlicking", "cunts", "cyalis", "cyberfuc", "cyberfuck", "cyberfucked", "cyberfucker", "cyberfuckers", "cyberfucking", "d1ck", "damn", "dick", "dickhead", "dildo", "dildos", "dink", "dinks", "dirsa", "dlck", "dog-fucker", "doggin", "dogging", "donkeyribber", "doosh", "duche", "dyke", "ejaculate", "ejaculated", "ejaculates", "ejaculating", "ejaculatings", "ejaculation", "ejakulate", "f u c k", "f u c k e r", "f4nny", "fag", "fagging", "faggitt", "faggot", "faggs", "fagot", "fagots", "fags", "fanny", "fannyflaps", "fannyfucker", "fanyy", "fatass", "fcuk", "fcuker", "fcuking", "feck", "fecker", "felching", "fellate", "fellatio", "fingerfuck", "fingerfucked", "fingerfucker", "fingerfuckers", "fingerfucking", "fingerfucks", "fistfuck", "fistfucked", "fistfucker", "fistfuckers", "fistfucking", "fistfuckings", "fistfucks", "flange", "fook", "fooker", "fuck", "fucka", "fucked", "fucker", "fuckers", "fuckhead", "fuckheads", "fuckin", "fucking", "fuckings", "fuckingshitmotherfucker", "fuckme", "fucks", "fuckwhit", "fuckwit", "fudge packer", "fudgepacker", "fuk", "fuker", "fukker", "fukkin", "fuks", "fukwhit", "fukwit", "fux", "fux0r", "f_u_c_k", "gangbang", "gangbanged", "gangbangs", "gaylord", "gaysex", "goatse", "God", "god-dam", "god-damned", "goddamn", "goddamned", "hardcoresex", "hell", "heshe", "hoar", "hoare", "hoer", "homo", "hore", "horniest", "horny", "hotsex", "jack-off", "jackoff", "jap", "jerk-off", "jism", "jiz", "jizm", "jizz", "kawk", "knob", "knobead", "knobed", "knobend", "knobhead", "knobjocky", "knobjokey", "kock", "kondum", "kondums", "kum", "kummer", "kumming", "kums", "kunilingus", "l3i+ch", "l3itch", "labia", "lmfao", "lust", "lusting", "m0f0", "m0fo", "m45terbate", "ma5terb8", "ma5terbate", "masochist", "master-bate", "masterb8", "masterbat*", "masterbat3", "masterbate", "masterbation", "masterbations", "masturbate", "mo-fo", "mof0", "mofo", "mothafuck", "mothafucka", "mothafuckas", "mothafuckaz", "mothafucked", "mothafucker", "mothafuckers", "mothafuckin", "mothafucking", "mothafuckings", "mothafucks", "mother fucker", "motherfuck", "motherfucked", "motherfucker", "motherfuckers", "motherfuckin", "motherfucking", "motherfuckings", "motherfuckka", "motherfucks", "muff", "mutha", "muthafecker", "muthafuckker", "muther", "mutherfucker", "n1gga", "n1gger", "nazi", "nigg3r", "nigg4h", "nigga", "niggah", "niggas", "niggaz", "nigger", "niggers", "nob", "nob jokey", "nobhead", "nobjocky", "nobjokey", "numbnuts", "nutsack", "orgasim", "orgasims", "orgasm", "orgasms", "p0rn", "pawn", "pecker", "penis", "penisfucker", "phonesex", "phuck", "phuk", "phuked", "phuking", "phukked", "phukking", "phuks", "phuq", "pigfucker", "pimpis", "piss", "pissed", "pisser", "pissers", "pisses", "pissflaps", "pissin", "pissing", "pissoff", "poop", "porn", "porno", "pornography", "pornos", "prick", "pricks", "pron", "pube", "pusse", "pussi", "pussies", "pussy", "pussys", "rectum", "retard", "rimjaw", "rimming", "s hit", "s.o.b.", "sadist", "schlong", "screwing", "scroat", "scrote", "scrotum", "semen", "sex", "sh!+", "sh!t", "sh1t", "shag", "shagger", "shaggin", "shagging", "shemale", "shi+", "shit", "shitdick", "shite", "shited", "shitey", "shitfuck", "shitfull", "shithead", "shiting", "shitings", "shits", "shitted", "shitter", "shitters", "shitting", "shittings", "shitty", "skank", "slut", "sluts", "smegma", "smut", "snatch", "son-of-a-bitch", "spac", "spunk", "s_h_i_t", "t1tt1e5", "t1tties", "teets", "teez", "testical", "testicle", "tit", "titfuck", "tits", "titt", "tittie5", "tittiefucker", "titties", "tittyfuck", "tittywank", "titwank", "tosser", "turd", "tw4t", "twat", "twathead", "twatty", "twunt", "twunter", "v14gra", "v1gra", "vagina", "viagra", "vulva", "w00se", "wang", "wank", "wanker", "wanky", "whoar", "whore", "willies", "willy", "xrated", "xxx"]
bad_dict = {}
for w in bad_word:
    bad_dict[w] = 1


def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
       # print(values.shape)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values, 
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), 
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    
    return logits


temperature = 1 #2.2
top_k = 50        #50
top_p = 0.95
device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_response(prompts, prefix_sentences, args):
    openai.api_key = args.api
    with torch.no_grad():
        sentences = []
        # output_sentences = [tokenizer.encode(x, add_prefix_space=True) for x in output_sentences_string]
        # prompt = [tokenizer.encode(x, add_prefix_space=True) for x in first_input_string]
        for i in range(len(prompts)):
            
            #total_string  = "There is office in the following response:" + output_sentences_string[i]
            # total_string  = "Make the following response full of office:" + output_sentences_string[i]
            # total_string = prompts[i] + prefix_sentences[i]
            # sentences.append(f"{total_string}\n\n")
            sentences.append(f"{prompts[i]}\n\nHuman: {prefix_sentences[i]}")
        reply_string = []

        response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=sentences,
                temperature=0,
                max_tokens=40,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
                stop=[" Human:", " AI:"]
                )
        for i in range(len(sentences)):
            reply_string.append(response['choices'][i]['text'])
        
        for i in range(len(reply_string)):
            reply_string[i] = [reply_string[i].strip()]

        if args.setting == 2:
            sentences = []
            for i in range(len(prompts)):
                sentences.append(f"Human: {prefix_sentences[i]}\nAI: {reply_string[i]}")
            reply_string = []
            response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=sentences,
                    temperature=0,
                    max_tokens=40,
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
            

        

eps = 0.0000000001


# +
def train(model_train, inputs_id, mask, tokenizer, ll, args, batch_size):
    loss = 0
    inputs_id = inputs_id.to(device_0)
    
    sep = tokenizer.sep_token_id # 102

    position_ids = mask.long().cumsum(-1) - 1 #+ prev_input.shape[1]
    position_ids.masked_fill_(mask == 0, 1)
    position_ids = position_ids.to(device_0)
    
    mask = mask.to(device_0)

    if args.dependence is True:
        prev_input, past = model_train(inputs_id, past_key_values=None, attention_mask=mask, position_ids=position_ids, return_dict=False)
    else:
        past = None
    
    # inputs_id = inputs_id.to(device_1)
    # position_ids = position_ids.to(device_1)
    # mask = mask.to(device_1)
    # with torch.no_grad():
    #     prev_input, past_bot = model_2(inputs_id, past_key_values=None, attention_mask=mask, position_ids=position_ids)
    prev_input = torch.LongTensor([[sep]] * inputs_id.shape[0]).to(device_0) # (8, 1)
    


    temp_sentence = [[] for i in range(inputs_id.shape[0])]
    emotion_loss = [0 for i in range(inputs_id.shape[0])]
    # coherence_loss = [0 for i in range(inputs_id.shape[0])]
    # test_reward = [1 for i in range(inputs_id.shape[0])]
    # coherence_reward = [0 for i in range(inputs_id.shape[0])]

    append = torch.tensor([[1] for i in range(len(inputs_id))]).to(device_0)
    mask = torch.cat((mask, append), 1)

    position_ids = mask.long().cumsum(-1) - 1 #+ prev_input.shape[1]
    position_ids.masked_fill_(mask == 0, 1)
    position_ids = position_ids[:, -1].unsqueeze(-1)
    position_ids = position_ids.to(device_0)
    
    for i in range(40):
        prev_input = prev_input.to(device_0)
        logits, past = model_train(prev_input, past_key_values=past, attention_mask=mask, position_ids=position_ids, return_dict=False)
        prev_input = prev_input.to(device_1)
        position_ids = position_ids.to(device_1)

        # with torch.no_grad():
        #     logits_bot, past_bot = model_2(prev_input, past_key_values=past_bot, attention_mask=mask, position_ids=position_ids)
        mask = torch.cat((mask, append), 1)
        
        position_ids = mask.long().cumsum(-1) - 1 #+ prev_input.shape[1]
        position_ids.masked_fill_(mask == 0, 1)
        position_ids = position_ids[:, -1].unsqueeze(-1)
        position_ids = position_ids.to(device_0)
        
        logits = logits.squeeze(0).squeeze(1)
        logits = logits / temperature

        logits = torch.softmax(logits, dim=-1)
        # with torch.no_grad():
        #     logits_bot = torch.softmax(logits_bot.squeeze(0).squeeze(1) / temperature, dim=-1)
        prev_input = torch.multinomial(logits[:], num_samples=1) # [[],[],...]
#         prev_input = torch.argmax(logits[:], dim=-1).unsqueeze(-1) # [1,2,.....]
#         print(prev_input)

        probs = []
        for j in range(inputs_id.shape[0]):
            if i != 0 and temp_sentence[j][-1] == sep: 
                continue
#             probs.append(logits_bot[j][prev_input[j][0].item()].item())
# #             probs.append(np.log(logits_bot[j][prev_input[j][0].item()].item()))
#             test_reward[j] *= logits_bot[j][prev_input[j][0].item()].item()
        if len(probs) == 0:
            avg_prob = 0
        else:
            avg_prob = sum(probs) / len(probs)

        for j in range(inputs_id.shape[0]):
            if i != 0 and temp_sentence[j][-1] == sep: continue
            temp_loss = F.cross_entropy(logits[j].unsqueeze(0), prev_input.view(-1)[j].unsqueeze(0))
            # coherence_loss[j] += (logits_bot[j][prev_input[j][0].item()].item() - avg_prob) * temp_loss
#             coherence_loss[j] += (np.log(logits_bot[j][prev_input[j][0].item()].item()) - avg_prob) * temp_loss
#             coherence_reward[j] *= (logits_bot[j][prev_input[j][0].item()].item() - avg_prob) * temp_loss
            emotion_loss[j] += temp_loss

        if i == 0:
            for j in range(inputs_id.shape[0]):
                temp_sentence[j].append(prev_input[j].item())
            continue
        flag = 1
        
        for j in range(0, inputs_id.shape[0]):
            if temp_sentence[j][-1] != sep: 
                flag = 0
                temp_sentence[j].append(prev_input[j].item())
        if flag == 1: break
    decode_temp_sentence = [tokenizer.decode(x).replace(' ', '') for x in temp_sentence]
    
    first_input = list(inputs_id.cpu().detach().numpy())
    for j in range(inputs_id.shape[0]):
        l = ll[j]
        first_input[j] = first_input[j][-l:]
        np.append(first_input[j], [sep], axis=-1)
    inter_response = []
    print(decode_temp_sentence)
    print(first_input)


    if 'gpt' in args.inter:
        inter_response.extend(make_response(decode_temp_sentence, first_input, args))

    print(inter_response)
    # if 'google' in args.inter:
    #     #k = []
    #     for j in range(inputs_id.shape[0]):
    #         k.append([jack.daemonPredict(sentence=a[j].replace('<|endoftext|>', ''))])
    # if 'retrieve' in args.inter:
    #     ii = []
    #     for j in range(inputs_id.shape[0]): 
    #         # ii = [tokenizer.decode(x[:-1]) for x in first_input]
    #         ii.append([tokenizer.decode(first_input[j][:-1]), a[j].replace('<|endoftext|>', '')])
    #     rps = ret_model.get_response(ii)
    #     k.extend([[x] for x in rps])

    #test_score += avg_prob
    sent_input = []

    for j in range(inputs_id.shape[0]*len(args.inter)):
        l = ll[j%inputs_id.shape[0]]
        sent_input.append([tokenizer.decode(inputs_id[j%inputs_id.shape[0]][:l].tolist()), decode_temp_sentence[j%inputs_id.shape[0]], inter_response[j][0]])
    
    temp_score = []
    for sens in sent_input:
        sen = (sens[0] + sens[1] + sens[2]).replace('[SEP]', '').replace('[CLS]', '').replace(' ', '')
        temp_score.append(len(sens[2]))


    # temp_score = []

#-----------------emotion-----------------------------

    # for e in embans:
    #     temp_score.append(np.sum((e - emo_embed)**2))
       

    # score = [0 for i in range(len(temp_score) // len(args.inter))]

    # for j in range(len(temp_score) // len(args.inter)):
    #     for k in range(len(args.inter)):
    #         score[j] += temp_score[j + batch_size*k]
#----------------specific word-------------------------------------------
    # score = np.array([0 for w in range(inputs_id.shape[0])])
    # for j in range(inputs_id.shape[0]*len(args.inter)):
    #     for word in bad_dict.keys():
    #         if re.search(r"\b{}\b".format(word.lower()), k[j].lower().strip()):
    #             score[j%8] += 1

    score = np.array(temp_score) / len(args.inter)
    score = score - np.mean(score)

    # test_len = [len(s) for s in temp_sentence]
    # test_reward = [test_reward[i] ** (1/test_len[i]) for i in range(inputs_id.shape[0])]
    
    # coherence_reward = [coherence_reward[i] ** (1/test_len[i]) for i in range(inputs_id.shape[0])]

    for j in range(inputs_id.shape[0]):
        loss += (score[j]) * emotion_loss[j] #/ len(temp_sentence[j])
        # loss += coherence_loss[j] * args.ra #/ len(temp_sentence[j])
#         loss += test_reward[j] * emotion_loss[j] * args.ra
#         loss -= coherence_reward[j] * args.ra
        
    # test_reward = np.mean(test_reward)

    raise
    return loss, sum(temp_score)


# -

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/train-v4.tsv')
#     parser.add_argument("--val_data_path", type=str, default='/work/u7930486/data/traditional_corpus/valid-v4.tsv')
#     parser.add_argument("--writer", type=str, default="")
    parser.add_argument("--sample_per_batch", type=int, default=5)
    parser.add_argument("--save", type=str, default="prompt_davinci_length")
    parser.add_argument("--model", type=str, default="ckiplab/gpt2-base-chinese")
    parser.add_argument("--ra", type=float, default=50)
    parser.add_argument("--inter", type=str, default="gpt", nargs='+', required=True)
    parser.add_argument("--setting", type=int, default=1)
    parser.add_argument("--dependence", type=bool, default=True)
    parser.add_argument("--api", type=str, default=None)

    args = parser.parse_args()
    if args.api == None:
        print("You should enter your open ai api to use this model")
        raise

    os.makedirs(os.path.join('models', args.save), exist_ok=True)
    
    wandb.init(project=args.save, entity="chatbot")
    wandb.login()
    wandb.config.update(args)
    
    np.random.seed(100)
    torch.random.manual_seed(100)
    torch.cuda.manual_seed(100)
    model_train = GPT2LMHeadModel.from_pretrained('ckiplab/gpt2-base-chinese')
    # model_train.load_state_dict(torch.load(args.model))
    # model_2 = GPT2LMHeadModel.from_pretrained('ckiplab/gpt2-base-chinese')
    # model_2.load_state_dict(torch.load(args.model))
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    # special_tokens_dict = {'eos_token': '[EOS]'}
    # tokenizer.add_special_tokens(special_tokens_dict)


    # if 'gpt' in args.inter:
    #     model_bot = GPT2LMHeadModel.from_pretrained('ckiplab/gpt2-base-chinese')
    #     model_bot.load_state_dict(torch.load(args.model))
    #     model_bot.to(device_1)
    #     model_bot.eval()

    writer = SummaryWriter(os.path.join('runs', args.save))
    param_optimizer = list(model_train.named_parameters())
    no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = Adam(optimizer_grouped_parameters, 5e-6,
                     max_grad_norm=1.0)

    model_train.to(device_0)
    # model_2.to(device_1)
    # model_2.eval()
    batch_size = 8

    
    print("processing dataset...")
    dataset = ChineseDataset(args.data_path, tokenizer, maxline=400000)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     val_dataset = ChineseDataset(args.val_data_path, tokenizer)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    batch = 0
    temp_score = 0
#     loss = 0
   
    test_score = 0
    for global_step in range(1):
        model_train.train()
        for inputs_id, mask, label, token_type_ids, first_input, first_mask, ll, position_ids in tqdm(train_dataloader):
            batch += 1
            
            loss = 0
            for _ in range(args.sample_per_batch):
                batch_loss, score = train(model_train, first_input, first_mask, tokenizer, ll, args, batch_size)
                loss += batch_loss

                # test_score += avg_prob
                temp_score += score
            loss.backward()
            
            if batch % 20 == 0:
                writer.add_scalar('reward', temp_score/batch_size/20, batch)
                wandb.log({"reward": temp_score/batch_size/20}, batch)
                # writer.add_scalar('test_reward', test_score/20, batch)
                # wandb.log({"test_reward": test_score/20}, batch)
                writer.add_scalar('loss', loss, batch)
                wandb.log({"loss": loss}, batch)
                print("Reward:%.2f,    test:%.6f   "%(temp_score/batch_size/20))
                # test_score = 0
                temp_score = 0
            if batch % 4 == 0:
#                 loss.backward()
                optimizer.step()
                optimizer.zero_grad()  
#                 loss = 0
            if batch % 1000 == 0:
                torch.save(
                    {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                        for k, v in model_train.state_dict().items()},
                    os.path.join('./models', args.save, 
                            f'model-{batch}.pkl'))

if __name__ == "__main__":
    main()
