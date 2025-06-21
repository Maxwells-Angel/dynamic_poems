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
thresh = .3
branches_add = 1 
running_input = [None]
output = []

def sestina_brancher(running_input: str):
    running_base_input = tokenizer([running_input], return_tensors = "pt").to("mps")
    token_ids = running_base_input['input_ids']
    curr_position = token_ids.shape[1] # len of str != tokens
    
    if curr_position >= n_tokens: #base case - if you've finished the whole thing
        output.append(running_input)
        return 

    coin_flip = random.random()
    if coin_flip < thresh:
        max_len = curr_position+1
        branches = model.generate(**running_base_input, max_length = max_len) # TODO - figure out how to generate logits 
        for i in range(branches_add): 
            # selecting through best words from "branches"
            #print(branches)
            new_token = branches[:,-1] #highest ranking 
            new_word = tokenizer.decode(new_token)
            # TODO - to help with comprehensibility, implement small beam search, i.e., checking if generated next token/word has high probability 
            # of working with the token/word we know comes after it, if not, re-search
            # decode choice and add it to the string with space before - TODO, address some of the edge cases here
            new_input = running_input + new_word
            sestina_brancher(new_input)
    else: 
        new_token = sestina_tokens[0,curr_position]
        new_word = tokenizer.convert_ids_to_tokens(new_token.item()) #TODO -- figure out why the tokenizer is not decoding token id -- PROBLEM WITH BOS TOKEN WHERE WE'RE STARTING 
        new_input = running_input + new_word
        sestina_brancher(new_input)
        # call the LLM with the tokens we generated so far 

#sestina_brancher("")
#print(output)

example_string = "My name is "
token_string = tokenizer([example_string], return_tensors = "pt").to("mps")
generation = model.generate(**token_string, max_length = token_string['input_ids'].shape[1]+5, return_dict_in_generate=True, output_logits=True)
#print("Scores")
#print(generation.logits[0].shape) # [1,262144]
#print(generation.logits) # 1, 502303

results1 = generation.logits[-1].squeeze()
results2 = generation.logits[-2].squeeze()
sorted_results1 = torch.sort(results1, descending=True)
sorted_results2 = torch.sort(results2, descending=True)
#print(sorted_results.indices)

for tok in sorted_results1.indices[:4]:
    print(tok)
    print(tokenizer.convert_ids_to_tokens(tok.item()))

for tok in sorted_results2.indices[:4]:
    print(tok)
    print(tokenizer.convert_ids_to_tokens(tok.item()))

#print("Sequences?")
#print(generation.sequences)