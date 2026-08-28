import torch

# Hugging Face Transformers provides:
#
# AutoTokenizer:
#   Converts human-readable question/context text into the
#   numerical token representation expected by BERT.
#
# AutoModelForQuestionAnswering:
#   Loads a pretrained Transformer model configured for
#   extractive question answering.
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# ---------------------------------------------------------
# 1. CHOOSE A PRETRAINED QUESTION-ANSWERING MODEL
# ---------------------------------------------------------

# This is a BERT Large model that was fine-tuned on SQuAD
# for extractive question answering.
#
# Extractive QA means:
#
#   The model does NOT generate a completely new answer.
#
# Instead, it predicts:
#
#   where the answer STARTS in the supplied text
#   where the answer ENDS in the supplied text
#
# and extracts that span.
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"


# ---------------------------------------------------------
# 2. LOAD THE TOKENIZER
# ---------------------------------------------------------

# The tokenizer converts:
#
#     Question + Context
#
# into:
#
#     token IDs
#     attention masks
#     token type information
#
# that BERT can process.
tokenizer = AutoTokenizer.from_pretrained(model_name)


# ---------------------------------------------------------
# 3. LOAD THE PRETRAINED QA MODEL
# ---------------------------------------------------------

# Load the already-trained question-answering model.
#
# There is NO model training in this script.
#
# We are performing inference using a model that has
# previously been trained/fine-tuned.
model = AutoModelForQuestionAnswering.from_pretrained(model_name)


# ---------------------------------------------------------
# 4. DEFINE THE CONTEXT
# ---------------------------------------------------------

# The context is the source of information from which
# BERT must extract answers.
#
# Notice that this context tells us:
#
#   WHAT battle?
#       Battle of Gettysburg
#
#   WHEN?
#       1863
#
#   WHERE?
#       Pennsylvania
#
# But it says NOTHING about:
#
#   Who commanded the Confederate Army?
#
# That becomes important for the second question.
context = "The Battle of Gettysburg was fought in 1863 in Pennsylvania"


# ---------------------------------------------------------
# 5. DEFINE MULTIPLE QUESTIONS
# ---------------------------------------------------------

# Instead of asking one question, we now have a list
# containing two questions.
#
# Question 1:
#
#   Can be answered directly from the supplied context.
#
#   Expected answer:
#
#       "1863"
#
#
# Question 2:
#
#   CANNOT be answered from this context.
#
#   The context contains no information about the general
#   of the Confederate Army.
#
# IMPORTANT:
#
# This model was fine-tuned on the original SQuAD dataset,
# which mainly assumes that an answer exists somewhere
# inside the context.
#
# Therefore, an unanswerable question may still cause the
# model to select some span from the context.
#
# This is a very useful AI-testing scenario:
#
#     What does the system do when the required information
#     is missing?
questions = [
    "In what year was the Battle of Gettysburg fought?",
    "Who was the general of the Confederate Army at Gettysburg?",
]


# ---------------------------------------------------------
# 6. CREATE AN EMPTY LIST FOR THE RESULTS
# ---------------------------------------------------------

# We will store one dictionary for every question.
#
# Conceptually:
#
# [
#     {
#         "question": "...",
#         "answer": "...",
#         "relevance_score": ...
#     },
#
#     {
#         "question": "...",
#         "answer": "...",
#         "relevance_score": ...
#     }
# ]
answers = []


# ---------------------------------------------------------
# 7. PROCESS EACH QUESTION
# ---------------------------------------------------------

# Loop through the questions one at a time.
#
# First iteration:
#
#     question =
#     "In what year was the Battle of Gettysburg fought?"
#
# Second iteration:
#
#     question =
#     "Who was the general of the Confederate Army at Gettysburg?"
for question in questions:


    # -----------------------------------------------------
    # 8. TOKENIZE THE CURRENT QUESTION + CONTEXT
    # -----------------------------------------------------

    # encode_plus() combines the question and context
    # into the form expected by BERT.
    #
    # Conceptually:
    #
    # [CLS]
    # question
    # [SEP]
    # context
    # [SEP]
    #
    #
    # add_special_tokens=True:
    #
    # Add BERT's special tokens such as [CLS] and [SEP].
    #
    #
    # return_tensors="pt":
    #
    # Return PyTorch tensors.
    inputs = tokenizer.encode_plus(
        question, context, add_special_tokens=True, return_tensors="pt"
    )


    # -----------------------------------------------------
    # 9. RUN THE QUESTION + CONTEXT THROUGH BERT
    # -----------------------------------------------------

    # **inputs unpacks the dictionary returned by
    # the tokenizer.
    #
    # Conceptually:
    #
    # model(
    #     input_ids=...,
    #     attention_mask=...,
    #     token_type_ids=...
    # )
    #
    # The model produces a score for every token indicating:
    #
    #   "How likely is this token to START the answer?"
    #
    # and:
    #
    #   "How likely is this token to END the answer?"
    outputs = model(**inputs)


    # -----------------------------------------------------
    # 10. GET START LOGITS
    # -----------------------------------------------------

    # start_logits contains one raw score for every token.
    #
    # Example conceptually:
    #
    # Token              Start score
    #
    # the                    -2.1
    # battle                 -1.4
    # gettysburg             -0.9
    # was                    -1.2
    # fought                 -0.7
    # in                     -0.3
    # 1863                    8.7   <-- highest
    # in                     -1.1
    # pennsylvania           -1.8
    #
    # These scores are LOGITS.
    #
    # They are NOT probabilities.
    start_scores = outputs.start_logits


    # -----------------------------------------------------
    # 11. GET END LOGITS
    # -----------------------------------------------------

    # Same idea, but now every token gets a score answering:
    #
    # "How likely is this token to be the END
    #  of the answer?"
    end_scores = outputs.end_logits


    # -----------------------------------------------------
    # 12. FIND THE HIGHEST-SCORING START POSITION
    # -----------------------------------------------------

    # torch.argmax() returns the index of the largest score.
    #
    # If token position 18 has the highest start score:
    #
    #     answer_start = 18
    #
    #
    # IMPROVEMENT NOTE:
    #
    # start_scores is already a PyTorch tensor.
    #
    # Therefore:
    #
    #     torch.tensor(start_scores)
    #
    # is unnecessary.
    #
    # The original script is intentionally left unchanged.
    answer_start = torch.argmax(torch.tensor(start_scores))


    # -----------------------------------------------------
    # 13. FIND THE HIGHEST-SCORING END POSITION
    # -----------------------------------------------------

    # Select the token with the largest end score.
    #
    # +1 is added because Python slicing excludes
    # the final index.
    #
    # If the answer ends at position 18:
    #
    #     tokens[18:19]
    #
    # includes token 18.
    answer_end = torch.argmax(torch.tensor(end_scores)) + 1


    # -----------------------------------------------------
    # 14. EXTRACT THE ANSWER TEXT
    # -----------------------------------------------------

    # Work from the inside outward:
    #
    # inputs["input_ids"][0]
    #
    # gives all token IDs for the current:
    #
    #     question + context
    #
    #
    # [answer_start:answer_end]
    #
    # selects the predicted answer span.
    #
    #
    # convert_ids_to_tokens()
    #
    # converts numerical token IDs back to BERT tokens.
    #
    #
    # convert_tokens_to_string()
    #
    # combines those tokens into human-readable text.
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )


    # -----------------------------------------------------
    # 15. CALCULATE A "RELEVANCE SCORE"
    # -----------------------------------------------------

    # The script defines its own relevance score as:
    #
    #     start logit of selected start token
    #
    #                 +
    #
    #     end logit of selected end token
    #
    #
    # In other words:
    #
    # relevance_score =
    #
    #     best start score + best end score
    #
    #
    # start_scores has shape approximately:
    #
    #     [1, number_of_tokens]
    #
    # Therefore:
    #
    # start_scores[0][answer_start]
    #
    # selects the start logit for the chosen token.
    #
    #
    # .item()
    #
    # converts the one-value PyTorch tensor into an
    # ordinary Python number.
    relevance_score = (
        start_scores[0][answer_start].item() + end_scores[0][answer_end - 1].item()
    )


    # IMPORTANT:
    #
    # "relevance_score" is a name chosen by this script.
    #
    # It is NOT a probability.
    #
    # It does NOT mean:
    #
    #     "The answer is 94% relevant"
    #
    # or:
    #
    #     "The model is 94% confident."
    #
    # It is simply:
    #
    #     selected start logit + selected end logit
    #
    # Larger values mean the model gave larger raw scores
    # to the selected span.
    #
    #
    # IMPROVEMENT / TESTING NOTE:
    #
    # Comparing raw logits between DIFFERENT questions
    # should be treated cautiously.
    #
    # Each question creates a different model input, so
    # these raw scores are not calibrated probabilities
    # that can automatically be compared as absolute
    # measures of answer correctness.


    # -----------------------------------------------------
    # 16. STORE THE QUESTION, ANSWER, AND SCORE
    # -----------------------------------------------------

    # Append one dictionary to the answers list.
    #
    # Example:
    #
    # {
    #     "question":
    #         "In what year ...?",
    #
    #     "answer":
    #         "1863",
    #
    #     "relevance_score":
    #         14.72
    # }
    answers.append(
        {"question": question, "answer": answer, "relevance_score": relevance_score}
    )


# ---------------------------------------------------------
# 17. SORT RESULTS BY RELEVANCE SCORE
# ---------------------------------------------------------

# sorted() creates a sorted version of the answers list.
#
# key=lambda x: x["relevance_score"]
#
# tells Python:
#
#     "Use the relevance_score field as the value
#      by which each dictionary should be sorted."
#
#
# reverse=True means:
#
#     largest score first
#     smallest score last
#
#
# Example:
#
# Before:
#
# [
#     {"score": 8.3},
#     {"score": 17.1}
# ]
#
# After:
#
# [
#     {"score": 17.1},
#     {"score": 8.3}
# ]
answers = sorted(answers, key=lambda x: x["relevance_score"], reverse=True)


# ---------------------------------------------------------
# 18. PRINT THE SORTED RESULTS
# ---------------------------------------------------------

# Iterate through the sorted results.
for answer in answers:


    # Print the original question.
    print("Question:", answer["question"])


    # Print the span BERT extracted as the answer.
    print("Answer:", answer["answer"])


    # Print the custom start-logit + end-logit score.
    print("Relevance Score:", answer["relevance_score"])


    # Print an empty line to visually separate results.
    print()