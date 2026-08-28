import torch

# Hugging Face Transformers provides pretrained tokenizers
# and pretrained Transformer models.
#
# AutoTokenizer:
#   Loads the tokenizer associated with a pretrained model.
#
# AutoModelForQuestionAnswering:
#   Loads a model specifically configured for
#   extractive question answering.
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# ---------------------------------------------------------
# 1. CHOOSE A PRETRAINED QUESTION-ANSWERING MODEL
# ---------------------------------------------------------

# This model is based on BERT Large.
#
# It was additionally fine-tuned on the SQuAD
# question-answering dataset.
#
# "uncased":
#   Text is treated without preserving capitalization
#   differences in the same way a cased model would.
#
# "whole-word-masking":
#   Refers to the masking strategy used during
#   BERT pretraining.
#
# "finetuned-squad":
#   The model was further trained for extractive
#   question answering using SQuAD.
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"


# ---------------------------------------------------------
# 2. LOAD THE TOKENIZER
# ---------------------------------------------------------

# The tokenizer converts human-readable text into the
# numerical token IDs expected by the BERT model.
#
# Conceptually:
#
#     "Gettysburg was fought in 1863"
#
#              ↓ tokenizer
#
#     tokens
#
#              ↓
#
#     token IDs
#
#              ↓
#
#     numbers that BERT can process
tokenizer = AutoTokenizer.from_pretrained(model_name)


# ---------------------------------------------------------
# 3. LOAD THE PRETRAINED QUESTION-ANSWERING MODEL
# ---------------------------------------------------------

# This downloads/loads the pretrained BERT model with
# a question-answering output head.
#
# The model has already learned:
#
# 1. General language representations during BERT pretraining
#
# 2. How to locate answer spans during question-answering
#    fine-tuning
#
# We are NOT training the model in this script.
#
# We are using an already-trained model for inference.
model = AutoModelForQuestionAnswering.from_pretrained(model_name)


# ---------------------------------------------------------
# 4. DEFINE THE CONTEXT
# ---------------------------------------------------------

# Extractive QA requires a piece of text containing
# the information from which the answer should be extracted.
#
# This is called the CONTEXT.
context = "The Battle of Gettysburg was fought in 1863 in Pennsylvania"


# ---------------------------------------------------------
# 5. DEFINE THE QUESTION
# ---------------------------------------------------------

# This is the question we want the model to answer
# using only the supplied context.
question = "In what year was the Battle of Gettysburg fought?"


# ---------------------------------------------------------
# 6. TOKENIZE THE QUESTION AND CONTEXT
# ---------------------------------------------------------

# encode_plus() converts the question and context into
# the numerical representation expected by BERT.
#
# We provide:
#
#     question
#     context
#
# together because BERT needs to examine them jointly.
#
#
# add_special_tokens=True:
#
# Adds special BERT tokens such as:
#
#     [CLS]
#     [SEP]
#
# Conceptually, the combined sequence looks something like:
#
# [CLS]
# In
# what
# year
# was
# the
# Battle
# of
# Gettysburg
# fought
# ?
# [SEP]
# The
# Battle
# of
# Gettysburg
# was
# fought
# in
# 1863
# in
# Pennsylvania
# [SEP]
#
#
# return_tensors="pt":
#
# Return PyTorch tensors.
#
# "pt" = PyTorch.
#
# The resulting "inputs" dictionary contains information
# such as:
#
#     input_ids
#     token_type_ids
#     attention_mask
#
# depending on what the tokenizer/model uses.
inputs = tokenizer.encode_plus(
    question,
    context,
    add_special_tokens=True,
    return_tensors="pt"
)


# ---------------------------------------------------------
# 7. PASS THE TOKENIZED INPUT THROUGH BERT
# ---------------------------------------------------------

# **inputs "unpacks" the dictionary.
#
# So conceptually:
#
# model(**inputs)
#
# becomes something similar to:
#
# model(
#     input_ids=inputs["input_ids"],
#     attention_mask=inputs["attention_mask"],
#     token_type_ids=inputs["token_type_ids"]
# )
#
# The exact supplied fields depend on the tokenizer.
#
#
# The model processes the complete question + context
# sequence and produces question-answering outputs.
outputs = model(**inputs)


# ---------------------------------------------------------
# 8. GET START-POSITION SCORES
# ---------------------------------------------------------

# For extractive question answering, BERT tries to answer
# TWO questions:
#
# 1. At which token does the answer START?
#
# 2. At which token does the answer END?
#
#
# start_logits contains a score for EVERY token indicating:
#
# "How likely is this token to be the beginning
#  of the answer?"
#
# Example conceptually:
#
# Token            Start score
#
# The                 -2.1
# Battle              -1.8
# Gettysburg          -2.3
# was                 -1.5
# fought              -1.0
# in                  -0.8
# 1863                 7.4   <-- highest
# in                  -1.2
# Pennsylvania        -2.0
#
# These are LOGITS, not probabilities.
start_scores = outputs.start_logits


# ---------------------------------------------------------
# 9. GET END-POSITION SCORES
# ---------------------------------------------------------

# end_logits similarly contains one score per token:
#
# "How likely is this token to be the END of the answer?"
#
# For a one-token answer such as "1863", the same token
# may receive the highest start and end scores.
end_scores = outputs.end_logits


# ---------------------------------------------------------
# 10. FIND THE MOST LIKELY START TOKEN
# ---------------------------------------------------------

# torch.argmax() returns the INDEX containing
# the largest value.
#
# So if the token "1863" has the highest start score,
# answer_start becomes the token position of "1863".
#
#
# IMPROVEMENT NOTE:
#
# start_scores is already a PyTorch tensor.
#
# Wrapping it again with:
#
#     torch.tensor(start_scores)
#
# is unnecessary.
#
# The original script is left unchanged.
answer_start = torch.argmax(
    torch.tensor(start_scores)
)


# ---------------------------------------------------------
# 11. FIND THE MOST LIKELY END TOKEN
# ---------------------------------------------------------

# Find the token having the highest end score.
#
# +1 is added because Python slicing excludes the
# ending index.
#
# Example:
#
# If:
#
#     answer_start = 18
#     answer_end   = 18
#
# then:
#
#     tokens[18:18]
#
# would return nothing.
#
# By adding 1:
#
#     tokens[18:19]
#
# includes token 18.
answer_end = torch.argmax(
    torch.tensor(end_scores)
) + 1


# ---------------------------------------------------------
# 12. EXTRACT THE ANSWER TOKENS
# ---------------------------------------------------------

# This line performs several steps.
#
# Let's break it down from the inside out.
answer = tokenizer.convert_tokens_to_string(

    # -----------------------------------------------------
    # STEP A: CONVERT TOKEN IDs BACK INTO TOKENS
    # -----------------------------------------------------
    #
    # inputs["input_ids"][0] contains the numerical IDs
    # representing the question + context.
    #
    # Example conceptually:
    #
    # [101, 1999, 2054, ..., 2560, ..., 102]
    #
    #
    # [answer_start:answer_end]
    #
    # selects only the token IDs belonging to the
    # predicted answer span.
    #
    #
    # convert_ids_to_tokens():
    #
    # converts those numeric IDs back into BERT tokens.
    #
    # Example:
    #
    #     [2560]
    #
    # could become:
    #
    #     ["1863"]
    tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0][answer_start:answer_end]
    )
)


# ---------------------------------------------------------
# 13. PRINT THE EXTRACTED ANSWER
# ---------------------------------------------------------

# For this example, we expect something similar to:
#
#     1863
print(answer)