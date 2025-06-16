# working with a sestina 
# to keep the end words anchored 
# So far, simple example of working with transformers to generate some output 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, einops, random

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

sestina = """
September rain falls on the house.
In the failing light, the old grandmother
sits in the kitchen with the child
beside the Little Marvel Stove,
reading the jokes from the almanac,
laughing and talking to hide her tears.
"""

masked_sestina = """
September rain falls on the house.
In the failing light, the old grandmother
sits in the kitchen with the child
beside the Little Marvel Stove,
reading the jokes from the almanac,
laughing and talking to hide her tears.
"""
base_input = tokenizer([masked_sestina], return_tensors="pt").to("mps")
sestina_tokens = base_input['input_ids']
n_tokens = sestina_tokens.shape[1]
thresh = .1
branches_add = 1 
running_input = [None]
output = [None]

def sestina_brancher(running_input: str):
    running_base_input = tokenizer([running_input], return_tensors = "pt").to("mps")
    token_ids = running_base_input['input_ids']
    curr_position = len(token_ids) # len of str != tokens
    print(curr_position, running_input)
    
    if curr_position == n_tokens:
        output.append(running_input)
        return 

    coin_flip = random.random()
    if coin_flip <= thresh:
        max_len = curr_position+1
        print("max length is",max_len)
        branches = model.generate(**running_base_input, max_length = max_len) # RETURN SCORES 
        for i in range(branches_add): 
            print("coinflip is initiated")
            # selecting through best words from "branches"
            new_token = branches[:,0]
            new_word = tokenizer.decode(new_token)
            # decode choice and add it to the string with space before - TODO, address some of the edge cases here
            new_input = running_input + new_word
            sestina_brancher(new_input)
    else: 
        new_token = sestina_tokens[0,curr_position]
        print(new_token.item())
        new_word = tokenizer.convert_ids_to_tokens(new_token.item()) #TODO -- figure out why the tokenizer is not decoding token id -- PROBLEM WITH BOS TOKEN WHERE WE'RE STARTING 
        print("the new word is", new_word)
        new_input = running_input + new_word
        sestina_brancher(new_input)
        # call the LLM with the tokens we generated so far 



test_word = tokenizer.convert_ids_to_tokens(1229)
print(test_word)



#sestina_brancher("September")
#print(output)

