import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

# Qwen2.5-1.5B-Instruct is an instruction-tuned language model.
#
# Unlike the base GPT-2 model used previously, this model
# has been trained to follow instructions and respond in
# a question/answer or conversational format.
model_name = "Qwen/Qwen2.5-1.5B-Instruct"


# This file was produced by our earlier preprocessing script.
#
# It should contain data conceptually like:
#
# [
#     {
#         "text": "Cleaned document text..."
#     }
# ]
text_file = "text_preprocessed.json"


# Maximum number of NEW tokens that the model can generate
# for the answer.
#
# This does NOT include the tokens belonging to:
#
#     system instruction
#     document
#     question
#
# Those are input tokens.
max_new_tokens = 150


# ---------------------------------------------------------
# 2. SELECT CPU OR GPU
# ---------------------------------------------------------

# If CUDA is available, use the NVIDIA GPU.
#
# Otherwise fall back to CPU.
#
# Therefore:
#
#     CUDA available -> cuda
#     CUDA unavailable -> cpu
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ---------------------------------------------------------
# 3. LOAD THE PREPROCESSED DOCUMENT
# ---------------------------------------------------------

# Open the JSON created by the previous preprocessing stage.
with open(text_file, "r", encoding="utf-8") as f:
    data = json.load(f)


# The JSON can potentially contain several scraped documents.
#
# Each object is expected to contain:
#
#     {
#         "text": "..."
#     }
#
# Join all of their text fields into one document.
#
# If there is only one object, this simply gives us
# that one object's cleaned text.
document_text = "\n".join(
    item["text"] for item in data
)


# ---------------------------------------------------------
# 4. DEFINE THE QUESTION
# ---------------------------------------------------------

question = (
    "How has the understanding of the term religion "
    "evolved over time?"
)


# ---------------------------------------------------------
# 5. LOAD THE QWEN TOKENIZER
# ---------------------------------------------------------

print("Loading tokenizer...")

# The tokenizer converts human-readable text into
# numerical token IDs that Qwen can understand.
tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

print("Tokenizer loaded.")


# ---------------------------------------------------------
# 6. LOAD THE QWEN MODEL
# ---------------------------------------------------------

print("Loading model...")


# AutoModelForCausalLM loads Qwen as a causal language model.
#
# "Causal language model" means the model generates text
# one token at a time based on everything that came before.
#
#
# torch_dtype="auto":
#
# Let Hugging Face use the model's preferred numeric type.
#
#
# attn_implementation="eager":
#
# We explicitly use eager attention because the optimized
# SDPA CUDA path caused a device-side assertion in our
# environment during the earlier GPT-2 experiment.
#
# This trades some performance for compatibility/stability.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    attn_implementation="eager",
)


# Move the model onto:
#
#     CUDA GPU
#
# or:
#
#     CPU
#
# depending on the device selected earlier.
model = model.to(device)


# Put the model into inference/evaluation mode.
#
# model.eval() does NOT generate anything itself.
#
# It simply tells PyTorch that this model is being used
# for inference rather than training.
model.eval()

print("Model loaded.")


# ---------------------------------------------------------
# 7. CREATE THE CHAT MESSAGES
# ---------------------------------------------------------

# Instruction-tuned models such as Qwen are designed to
# work with structured conversational messages.
#
# We use two roles:
#
#     system
#     user
#
#
# SYSTEM:
#
# Defines the behavior we want from the model.
#
# USER:
#
# Supplies the actual document and question.
#
#
# The grounding instruction is especially important:
#
#     "Answer using only the supplied document."
#
# This explicitly asks the model NOT to fill gaps using
# outside knowledge.
messages = [
    {
        "role": "system",
        "content": (
            "You are a document question-answering assistant. "
            "Answer the question using only information contained "
            "in the supplied document. "
            "Do not use outside knowledge. "
            "If the document does not contain enough information "
            "to answer the question, respond exactly with: "
            "'The answer is not available in the document.' "
            "Keep the answer concise and directly relevant "
            "to the question."
        ),
    },
    {
        "role": "user",
        "content": (
            f"Document:\n"
            f"{document_text}\n\n"
            f"Question:\n"
            f"{question}"
        ),
    },
]


# ---------------------------------------------------------
# 8. APPLY QWEN'S CHAT TEMPLATE
# ---------------------------------------------------------

# Different instruction-tuned models expect special tokens
# and formatting around:
#
#     system messages
#     user messages
#     assistant messages
#
# We should NOT manually guess this formatting.
#
# apply_chat_template() uses the format expected by Qwen.
#
#
# tokenize=False:
#
# Return formatted TEXT first rather than token IDs.
#
#
# add_generation_prompt=True:
#
# Add the marker that tells Qwen:
#
#     "The assistant's response should start here."
#
#
# Conceptually, the resulting prompt becomes something like:
#
# SYSTEM:
# Answer only from the document...
#
# USER:
# Document:
# <document>
#
# Question:
# ...
#
# ASSISTANT:
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)


# ---------------------------------------------------------
# 9. TOKENIZE THE COMPLETE PROMPT
# ---------------------------------------------------------

# Convert:
#
#     system instruction
#     +
#     document
#     +
#     question
#
# into PyTorch tensors.
#
#
# return_tensors="pt":
#
# "pt" means PyTorch.
#
# The result contains things such as:
#
#     input_ids
#     attention_mask
inputs = tokenizer(
    prompt,
    return_tensors="pt",
)


# Move every tensor in the tokenizer output onto
# the same device as the model.
#
# For example:
#
#     CPU -> CUDA
#
# if we're using the GPU.
inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# ---------------------------------------------------------
# 10. INSPECT THE INPUT SIZE
# ---------------------------------------------------------

# Number of tokens actually being passed into Qwen.
input_token_count = inputs["input_ids"].shape[1]


# Qwen exposes its maximum positional/context size
# through the model configuration.
#
# For this model this should be vastly larger than the
# ~900-token document we were working with previously.
model_context_length = model.config.max_position_embeddings


print()
print("Document characters:", len(document_text))
print("Input tokens:", input_token_count)
print("Requested new tokens:", max_new_tokens)
print("Model context length:", model_context_length)


# ---------------------------------------------------------
# 11. CHECK THAT INPUT + OUTPUT FITS
# ---------------------------------------------------------

# The complete sequence must fit within the model's
# context window:
#
#     input tokens
#           +
#     generated tokens
#           <=
#     model context length
#
#
# Example:
#
#     1,100 input tokens
#     +
#       150 generated tokens
#     =
#     1,250 total
#
# which would easily fit into a much larger context window.
assert (
    input_token_count + max_new_tokens
    <= model_context_length
), (
    f"Context window exceeded: "
    f"{input_token_count} input tokens + "
    f"{max_new_tokens} generated tokens > "
    f"{model_context_length}"
)


# ---------------------------------------------------------
# 12. GENERATE THE ANSWER
# ---------------------------------------------------------

print()
print("Generating answer...")


# torch.no_grad() disables gradient calculations.
#
# We are doing:
#
#     inference
#
# not:
#
#     training
#
# so gradients and backpropagation are unnecessary.
with torch.no_grad():

    outputs = model.generate(

        # Pass all tokenizer-created model inputs:
        #
        #     input_ids
        #     attention_mask
        #
        # into generate().
        **inputs,


        # Generate at most this number of NEW tokens.
        max_new_tokens=max_new_tokens,


        # Deterministic generation.
        #
        # This means the model does not randomly sample
        # from possible next tokens.
        #
        # This is particularly useful while testing because:
        #
        # same document
        # +
        # same question
        # +
        # same model
        #
        # should normally produce the same answer.
        do_sample=False,


        # Mildly discourage repetitive output.
        repetition_penalty=1.1,


        # Qwen already has suitable end-of-sequence tokens.
        #
        # pad_token_id is supplied explicitly to avoid
        # generation warnings in some configurations.
        pad_token_id=tokenizer.eos_token_id,
    )


print("Text generation complete.")


# ---------------------------------------------------------
# 13. REMOVE THE ORIGINAL PROMPT TOKENS
# ---------------------------------------------------------

# model.generate() returns:
#
#     original input tokens
#              +
#     generated answer tokens
#
#
# Suppose:
#
#     input length = 1000
#
# and Qwen generates:
#
#     50 new tokens
#
# Then outputs[0] contains:
#
#     1050 tokens
#
#
# We only want the 50 NEW answer tokens.
generated_token_ids = outputs[0][input_token_count:]


# ---------------------------------------------------------
# 14. CONVERT ANSWER TOKEN IDs BACK TO TEXT
# ---------------------------------------------------------

# Decode the generated token IDs into human-readable text.
#
# skip_special_tokens=True removes internal control tokens
# used by the model.
answer = tokenizer.decode(
    generated_token_ids,
    skip_special_tokens=True,
).strip()


# ---------------------------------------------------------
# 15. PRINT THE RESULT
# ---------------------------------------------------------

print()
print("Question:")
print(question)

print()

print("Answer:")
print(answer)