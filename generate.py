"""Generate text from a pre-trained model."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
# Download configuration from huggingface.co and cache.

import data

def generate(model_dir = 'models/gpt2_homer'):
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

    PADDING_TEXT = """The quarrel between Agamemnon and Achilles—Achilles withdraws from the war, and sends his mother Thetis to ask Jove to help the Trojans—Scene between Jove and Juno on Olympus.
    Sing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans.
    Many a brave soul did it send hurrying down to Hades, and many a hero did it yield a prey to dogs and vultures, for so were the counsels of Jove fulfilled from the day on which the son of Atreus, king of men, and great Achilles, first fell out with one another.
    And which of the gods was it that set them on to quarrel?
    It was the son of Jove and Leto; for he was angry with the king and sent a pestilence upon the host to plague the people, because the son of Atreus had dishonoured Chryses his priest.
    Now Chryses had come to the ships of the Achaeans to free his daughter, and had brought with him a great ransom: moreover he bore in his hand the sceptre of Apollo wreathed with a suppliant’s wreath, and he besought the Achaeans, but most of all the two sons of Atreus, who were their chiefs.
    “Sons of Atreus,” he cried, “and all other Achaeans, may the gods who dwell in Olympus grant you to sack the city of Priam, and to reach your homes in safety; but free my daughter, and accept a ransom for her, in reverence to Apollo, son of Jove.”  On this the rest of the Achaeans with one voice were for respecting the priest and taking the ransom that he offered; but not so Agamemnon, who spoke fiercely to him and sent him roughly away.
    “Old man,” said he, “let me not find you tarrying about our ships, nor yet coming hereafter.
    Your sceptre of the god and your wreath shall profit you nothing. <eod> </s> <eos>"""

    prompt = "I will not free her."
    inputs = tokenizer(PADDING_TEXT + prompt,
                    add_special_tokens=False,
                    return_tensors="pt")["input_ids"]
    prompt_length = len(tokenizer.decode(inputs[0]))
    outputs = model.generate(inputs,
                            max_length=1000,
                            do_sample=True,
                            top_p=0.95,
                            top_k=60)
    generated = prompt + tokenizer.decode(outputs[0])[prompt_length + 1:]
    print(generated)

    filename = 'output/gen.txt'
    textfile = open(filename, 'w+')
    textfile.write(generated)
    textfile.close()

if __name__ == '__main__':
    generate()