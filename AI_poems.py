# working with a sestina 
# to keep the end words anchored 
# So far, simple example of working with transformers to generate some output 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, einops, random, math, tqdm

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

sestina = """
September rain falls on the house.
In the failing light, the old grandmother
sits in the kitchen with the child
beside the Little Marvel Stove,
reading the jokes from the almanac,
laughing and talking to hide her tears.

She thinks that her equinoctial tears
and the rain that beats on the roof of the house
were both foretold by the almanac,
but only known to a grandmother.
The iron kettle sings on the stove.
She cuts some bread and says to the child,

It's time for tea now; but the child
is watching the teakettle's small hard tears
dance like mad on the hot black stove,
the way the rain must dance on the house.
Tidying up, the old grandmother
hangs up the clever almanac

on its string. Birdlike, the almanac
hovers half open above the child,
hovers above the old grandmother
and her teacup full of dark brown tears.
She shivers and says she thinks the house
feels chilly, and puts more wood in the stove.

It was to be, says the Marvel Stove.
I know what I know, says the almanac.
With crayons the child draws a rigid house
and a winding pathway. Then the child
puts in a man with buttons like tears
and shows it proudly to the grandmother.

But secretly, while the grandmother
busies herself about the stove,
the little moons fall down like tears
from between the pages of the almanac
into the flower bed the child
has carefully placed in the front of the house.

Time to plant tears, says the almanac.
The grandmother sings to the marvelous stove
and the child draws another inscrutable house.
"""

masked_sestina = """
September rain falls on the house.
In the failing light, the old grandmother
sits in the kitchen with the child
beside the Little Marvel Stove,
reading the jokes from the almanac,
laughing and talking to hide her tears.
"""
base_input = tokenizer([sestina], return_tensors="pt").to("mps")
sestina_tokens = base_input['input_ids']
n_tokens = sestina_tokens.shape[1]
branch_prob = .01
path_thresh = 17.0
branches_add = 100 
running_input = [None]
output = []

def sestina_brancher(running_input: torch.Tensor, running_score: list, count_branches: int):
    if (len(running_score) > 0 and sum(running_score)/len(running_score) < path_thresh):
        print("Branch killed")
        return False
    
    decoded_running_input = tokenizer.decode(running_input) # return the string version of the running input
    running_base_input = tokenizer([decoded_running_input], return_tensors = "pt").to("mps")
    token_ids = running_base_input['input_ids']
    curr_position = token_ids.shape[1] # len of str != tokens
    
    print(curr_position)
    if curr_position >= n_tokens: #base case - if you've finished the whole thing
        final_score = 0
        if (len(running_score) > 0):
            final_score = sum(running_score)/len(running_score)
        output.append({"text": running_input, "score": final_score, "branch": count_branches})
        return True

    coin_flip = random.random()
    if coin_flip < branch_prob:
        print("Branch", count_branches)
        max_len = curr_position+1

        generation = model.generate(**running_base_input, max_length = max_len, return_dict_in_generate = True, output_logits = True) # TODO - figure out how to generate logits 
        good_branch = True
        for i in range(branches_add): 
            # selecting through best words from "branches"
            logits = generation.logits[-1].squeeze().sort(descending = True)
            new_score = running_score + [logits.values[i].item()] # TODO 
            new_token = logits.indices[i].reshape(1)
            new_word = tokenizer.decode(new_token)
            # TODO - to help with comprehensibility, implement small beam search, i.e., checking if generated next token/word has high probability 
            # of working with the token/word we know comes after it, if not, re-search
            # decode choice and add it to the string with space before - TODO, address some of the edge cases here
            #new_input = running_input + new_word
            new_input = torch.cat((running_input, new_token))
            if not sestina_brancher(new_input, new_score, count_branches + 1): 
                good_branch = False         
        # if not good_branch:
        #     new_token = (sestina_tokens[0,curr_position]).reshape(1)
        #     new_input = torch.cat((running_input, new_token))   
        #     if not sestina_brancher(new_input, running_score, count_branches): return False
            

    else: 
        new_token = (sestina_tokens[0,curr_position]).reshape(1)
        new_word = tokenizer.convert_ids_to_tokens(new_token.item())  
        # new_input = running_input + new_word
        new_input = torch.cat((running_input, new_token))
        if not sestina_brancher(new_input, running_score, count_branches): return False



sestina_brancher(sestina_tokens[0,:2].squeeze(),[],0)
# print(output)

for dic in output: 
    print(tokenizer.decode(dic['text']))
    print(dic["score"])

print("Complete")
#print("Sequences?")
#print(generation.sequences)