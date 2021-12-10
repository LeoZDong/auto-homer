"""Generate text from a pre-trained model."""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Download configuration from huggingface.co and cache.

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def generate(out_file, model_dir='models/gpt2_homer', max_length=1000):
    if model_dir is None:
        print("Loading from default pretrained GPT-2!")
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    else:
        print("Loading from local model!")
        model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    PADDING_TEXT = """The quarrel between Agamemnon and Achilles—Achilles withdraws from the war, and sends his mother Thetis to ask Jove to help the Trojans—Scene between Jove and Juno on Olympus.
    Sing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans.
    Many a brave soul did it send hurrying down to Hades, and many a hero did it yield a prey to dogs and vultures, for so were the counsels of Jove fulfilled from the day on which the son of Atreus, king of men, and great Achilles, first fell out with one another.
    And which of the gods was it that set them on to quarrel?
    It was the son of Jove and Leto; for he was angry with the king and sent a pestilence upon the host to plague the people, because the son of Atreus had dishonoured Chryses his priest.
    Now Chryses had come to the ships of the Achaeans to free his daughter, and had brought with him a great ransom: moreover he bore in his hand the sceptre of Apollo wreathed with a suppliant’s wreath, and he besought the Achaeans, but most of all the two sons of Atreus, who were their chiefs.
    “Sons of Atreus,” he cried, “and all other Achaeans, may the gods who dwell in Olympus grant you to sack the city of Priam, and to reach your homes in safety; but free my daughter, and accept a ransom for her, in reverence to Apollo, son of Jove.”  On this the rest of the Achaeans with one voice were for respecting the priest and taking the ransom that he offered; but not so Agamemnon, who spoke fiercely to him and sent him roughly away.
    “Old man,” said he, “let me not find you tarrying about our ships, nor yet coming hereafter.
    Your sceptre of the god and your wreath shall profit you nothing. <eod> </s> <eos>"""
    # PADDING_TEXT = ""
    prompt = "Sing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans."
    inputs = tokenizer(PADDING_TEXT + prompt,
                       add_special_tokens=False,
                       return_tensors="pt")["input_ids"].to(device)
    prompt_length = len(tokenizer.decode(inputs[0]))
    outputs = model.generate(inputs,
                             max_length=1000,
                             do_sample=True,
                             top_p=0.95,
                             top_k=60,
                             no_repeat_ngram_size=2,
                             num_return_sequences=3,
                             early_stopping=True,
                             num_beams=5)
    # generate text until the output length (which includes the context length) reaches 50
    out_texts = []
    for i, output in enumerate(outputs):
        out_text = prompt + tokenizer.decode(
            output, skip_special_tokens=True)[prompt_length + 1:]
        print("{}: {}".format(i, out_text))
        out_texts.append(out_text)

    if out_file is not None:
        for i, out_text in enumerate(out_texts):
            filename = f'output/{out_file}_{i}.txt'
            textfile = open(filename, 'w+')
            textfile.write(out_text)
            textfile.close()

if __name__ == '__main__':
    generate(None, None)