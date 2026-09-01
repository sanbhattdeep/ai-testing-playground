import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

tokenizer_base = "gpt2"
model_base = "gpt2"

text_file = "text_preprocessed.json"

# Maximum number of NEW tokens GPT-2 is allowed to generate
# for the answer.
max_new_tokens = 50


# ---------------------------------------------------------
# 2. CHOOSE GPU OR CPU
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ---------------------------------------------------------
# 3. LOAD THE PREPROCESSED DOCUMENT
# ---------------------------------------------------------

# text_preprocessed.json was created by the previous
# preprocessing script.
#
# It should conceptually contain:
#
# [
#     {
#         "text": "Cleaned document text..."
#     }
# ]
#
# Unlike the previous version, we use json.load() so that
# we extract the actual document text rather than treating
# the JSON syntax itself as part of the document.
with open(text_file, "r", encoding="utf-8") as f:
    data = json.load(f)


# Combine all "text" fields into one document.
#
# If the file contains only one object, this simply gives us
# that object's text.
#
# If it contains several objects, their text is combined.
document_text = "\n".join(item["text"] for item in data)


# ---------------------------------------------------------
# 4. LOAD GPT-2 TOKENIZER
# ---------------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_base)

print("Tokenizer loaded.")


# ---------------------------------------------------------
# 5. LOAD GPT-2 MODEL
# ---------------------------------------------------------

print("Loading model...")

# model = AutoModelForCausalLM.from_pretrained(model_base).to(device)
model = AutoModelForCausalLM.from_pretrained(
    model_base,

    # Force the normal/manual attention implementation
    # instead of PyTorch's optimized SDPA CUDA path.
    attn_implementation="eager",
).to(device)

model.eval()
# Put the model into inference/evaluation mode.
model.eval()

print("Model loaded.")


# ---------------------------------------------------------
# 6. DEFINE THE QUESTION
# ---------------------------------------------------------

question = "How has the understanding of the term religion evolved over time?"


# ---------------------------------------------------------
# 7. CREATE DOCUMENT + QUESTION INPUT
# ---------------------------------------------------------

# We want GPT-2 to receive something conceptually like:
#
# Context:
# <contents of the document>
#
# Question:
# How has the understanding ...
#
# Answer:
#
#
# GPT-2 then continues generating text after "Answer:".
prefix = (
    "Read the following passage and answer the question "
    "using information from the passage.\n\n"
    "Passage:\n"
)

suffix = (
    f"\n\nQuestion: {question}\n"
    "Answer in one or two concise sentences based on the passage:\n"
)

# ---------------------------------------------------------
# 8. TOKENIZE THE FIXED PARTS OF THE PROMPT
# ---------------------------------------------------------

# Tokenize the text that appears before the document.
prefix_ids = tokenizer.encode(
    prefix,
    add_special_tokens=False
)

# Tokenize the question and "Answer:" portion.
suffix_ids = tokenizer.encode(
    suffix,
    add_special_tokens=False
)


# ---------------------------------------------------------
# 9. TOKENIZE THE DOCUMENT
# ---------------------------------------------------------

document_ids = tokenizer.encode(
    document_text,
    add_special_tokens=False
)


# ---------------------------------------------------------
# 10. WORK OUT HOW MUCH DOCUMENT TEXT CAN FIT
# ---------------------------------------------------------

# GPT-2 normally has a maximum context window of 1024 tokens.
#
# The complete sequence must fit:
#
#     prefix
#       +
#     document
#       +
#     question
#       +
#     generated answer
#
#
# We reserve max_new_tokens for GPT-2's answer.
model_max_length = model.config.n_positions


# Number of document tokens we can safely include:
max_document_tokens = (
    model_max_length
    - len(prefix_ids)
    - len(suffix_ids)
    - max_new_tokens
)


# If the document is too long, truncate ONLY the document.
#
# This is important because we do not want truncation
# to accidentally remove the question from the end.
document_ids = document_ids[:max_document_tokens]


# ---------------------------------------------------------
# 11. BUILD THE COMPLETE MODEL INPUT
# ---------------------------------------------------------

# Combine:
#
#     Context:
#       +
#     document tokens
#       +
#     Question + Answer:
#
input_ids = prefix_ids + document_ids + suffix_ids


# Convert the Python list of token IDs into a PyTorch tensor.
#
# Shape becomes:
#
#     [1, number_of_tokens]
#
# The first dimension represents a batch containing
# one input sequence.
input_ids = torch.tensor(
    [input_ids],
    dtype=torch.long
).to(device)


# Every position contains a real token, so the attention
# mask contains 1 for every token.
attention_mask = torch.ones_like(input_ids)


# ---------------------------------------------------------
# 12. SHOW HOW MUCH CONTEXT IS ACTUALLY BEING USED
# ---------------------------------------------------------

print("Document tokens:", len(document_ids))
print("Prompt tokens:", input_ids.shape[1])
print("Maximum GPT-2 context:", model_max_length)


# ---------------------------------------------------------
# 13. GENERATE THE ANSWER
# ---------------------------------------------------------

print("Generating text...")


# torch.no_grad() disables gradient calculation because
# we are doing inference, not training.
with torch.no_grad():

    output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,

    # We only need a relatively short answer.
    max_new_tokens=50,

    num_return_sequences=1,

    # Use sampling rather than always selecting the
    # single highest-scoring next token.
    do_sample=True,

    # Lower values make the model prefer more likely tokens.
    # 0.7 is usually a reasonable starting point.
    #temperature=0.7,

    # Only sample from tokens making up the top 90%
    # of the probability mass.
    #top_p=0.9,

    # Also restrict consideration to the 50 most
    # likely next tokens.
    #top_k=50,

    # Discourage repetition without completely banning
    # every repeated two-token sequence.
    repetition_penalty=1.15,

    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

print("Text generation complete.")


# ---------------------------------------------------------
# 14. EXTRACT ONLY THE NEWLY GENERATED TOKENS
# ---------------------------------------------------------

# model.generate() returns:
#
#     original prompt + generated answer
#
# We know exactly how many prompt tokens were supplied:
#
#     input_ids.shape[1]
#
# Therefore we slice from that position onward and keep
# only the newly generated tokens.
generated_token_ids = output[0][input_ids.shape[1] :]


# Convert the generated token IDs back into human-readable text.
generated_text = tokenizer.decode(
    generated_token_ids,
    skip_special_tokens=True
).strip()


# ---------------------------------------------------------
# 15. KEEP ONLY THE FIRST 30 WORDS
# ---------------------------------------------------------

# Preserve the behavior of the original script:
#
# print at most the first 30 whitespace-separated words.
answer = " ".join(generated_text.split()[:30])


# ---------------------------------------------------------
# 16. PRINT RESULT
# ---------------------------------------------------------

print()
print("Question:")
print(question)

print()
print("Answer:")
print(answer)