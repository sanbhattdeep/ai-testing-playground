# json is used to:
#
#     read text_preprocessed.json
#
# and:
#
#     write text_tokenized.json
import json


# AutoTokenizer lets Hugging Face automatically load the
# tokenizer associated with a particular pretrained model.
#
# In this script we later load:
#
#     bert-base-uncased
#
# so AutoTokenizer will load the tokenizer appropriate
# for BERT Base Uncased.
from transformers import AutoTokenizer


# ---------------------------------------------------------
# 1. DEFINE A FUNCTION TO TOKENIZE MULTIPLE TEXT DOCUMENTS
# ---------------------------------------------------------

# This function receives:
#
# tokenizer:
#     The Hugging Face tokenizer that converts text
#     into token IDs.
#
# texts:
#     The data loaded from text_preprocessed.json.
#
#     Based on the previous script, it should look
#     conceptually like:
#
#     [
#         {
#             "text": "First cleaned document..."
#         },
#         {
#             "text": "Second cleaned document..."
#         }
#     ]
#
# max_length:
#     Maximum sequence length used during tokenization.
#
#
# The function eventually returns a nested list:
#
# [
#     [
#         token_ids_for_chunk_1,
#         token_ids_for_chunk_2,
#         ...
#     ],
#
#     [
#         token_ids_for_another_document,
#         ...
#     ]
# ]
def tokenize_text(tokenizer, texts, max_length):


    # -----------------------------------------------------
    # 2. MAKE SURE THE TOKENIZER HAS A PAD TOKEN
    # -----------------------------------------------------

    # Padding means making sequences the same length.
    #
    # For example:
    #
    # Sequence A:
    #
    #     [101, 2009, 2003, 2204, 102]
    #
    # Sequence B:
    #
    #     [101, 7592, 102]
    #
    # If we want both sequences to have length 5,
    # Sequence B might become:
    #
    #     [101, 7592, 102, 0, 0]
    #
    # where the extra values represent padding.
    #
    #
    # Some tokenizers do not have a pad token defined.
    #
    # If that happens, this code adds:
    #
    #     [PAD]
    #
    # as the padding token.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})


    # IMPROVEMENT NOTE:
    #
    # bert-base-uncased already normally has a [PAD] token,
    # so this condition should generally be False for the
    # tokenizer used later in this script.
    #
    # This defensive check can still be useful if the function
    # is later reused with another tokenizer.


    # -----------------------------------------------------
    # 3. CREATE A CONTAINER FOR ALL TOKENIZED DOCUMENTS
    # -----------------------------------------------------

    # tokenized_texts will contain the final result.
    #
    # Each document can contain several chunks.
    #
    # Conceptually:
    #
    # tokenized_texts =
    #
    # [
    #     [
    #         chunk1_token_ids,
    #         chunk2_token_ids
    #     ],
    #
    #     [
    #         chunk1_token_ids,
    #         chunk2_token_ids,
    #         chunk3_token_ids
    #     ]
    # ]
    tokenized_texts = []


    # -----------------------------------------------------
    # 4. PROCESS EACH PREPROCESSED DOCUMENT
    # -----------------------------------------------------

    # "texts" is expected to contain dictionaries such as:
    #
    #     {
    #         "text": "The cleaned webpage content..."
    #     }
    #
    # During each loop:
    #
    #     text
    #
    # refers to one dictionary.
    for text in texts:


        # -------------------------------------------------
        # 5. SPLIT THE DOCUMENT INTO SMALLER CHUNKS
        # -------------------------------------------------

        # Extract the value stored under:
        #
        #     text["text"]
        #
        # and pass it into split_text().
        #
        # The reason for chunking is that BERT cannot
        # process an arbitrarily long sequence.
        #
        # BERT models normally have a maximum sequence
        # length of around 512 TOKENS.
        chunks = split_text(text["text"], max_length)


        # Create another list for the token IDs belonging
        # to the chunks of THIS particular document.
        chunk_token_ids = []


        # -------------------------------------------------
        # 6. TOKENIZE EACH CHUNK
        # -------------------------------------------------

        for chunk in chunks:


            # tokenizer.encode() performs several operations.
            #
            # It roughly transforms:
            #
            #     text
            #
            #       ↓
            #
            #     tokenizer tokens
            #
            #       ↓
            #
            #     token IDs
            #
            #
            # For example:
            #
            #     "The battle was fought"
            #
            # might conceptually become:
            #
            #     ["the", "battle", "was", "fought"]
            #
            # and then:
            #
            #     [1996, 2645, 2001, 4061]
            token_ids = tokenizer.encode(

                # The actual text chunk to tokenize.
                chunk,


                # -----------------------------------------
                # ADD BERT SPECIAL TOKENS
                # -----------------------------------------

                # BERT normally surrounds a single input
                # sequence with special tokens.
                #
                # Conceptually:
                #
                #     text
                #
                # becomes:
                #
                #     [CLS] text [SEP]
                #
                #
                # [CLS] indicates the start of the sequence.
                #
                # [SEP] indicates the end/separation
                # of a sequence.
                add_special_tokens=True,


                # -----------------------------------------
                # MAXIMUM TOKENIZED SEQUENCE LENGTH
                # -----------------------------------------

                # Tell the tokenizer that the resulting
                # sequence should have at most max_length
                # tokens.
                #
                # Here max_length will later be:
                #
                #     512
                max_length=max_length,


                # -----------------------------------------
                # TRUNCATE IF TOO LONG
                # -----------------------------------------

                # If tokenization produces more than
                # max_length tokens, remove tokens from
                # the end until the sequence fits.
                truncation=True,


                # -----------------------------------------
                # PAD SHORT SEQUENCES
                # -----------------------------------------

                # padding="max_length" means:
                #
                # Always produce exactly max_length
                # token positions.
                #
                # Suppose a chunk tokenizes to:
                #
                #     100 tokens
                #
                # but max_length=512.
                #
                # Padding is added until it reaches:
                #
                #     512 tokens
                #
                # Conceptually:
                #
                # real tokens
                #     +
                # [PAD] [PAD] [PAD] ...
                #     =
                # 512 positions
                padding="max_length",


                # -----------------------------------------
                # REQUEST AN ATTENTION MASK
                # -----------------------------------------

                # An attention mask normally tells the
                # Transformer which positions contain:
                #
                #     real tokens -> 1
                #
                # versus:
                #
                #     padding     -> 0
                #
                # Example:
                #
                # token_ids:
                #
                # [101, 1996, 2645, 102, 0, 0]
                #
                # attention_mask:
                #
                # [  1,    1,    1,   1, 0, 0]
                #
                return_attention_mask=True,
            )


            # IMPORTANT NOTE:
            #
            # tokenizer.encode() returns the token IDs.
            #
            # Although return_attention_mask=True appears
            # here, this script stores only the result
            # returned by encode(), which is the token-ID
            # sequence.
            #
            # Therefore text_tokenized.json contains token IDs,
            # not a separate attention mask.
            #
            # If both input_ids and attention_mask were needed
            # later, a tokenizer call returning a dictionary
            # would generally be more suitable.
            #
            # The original script is intentionally unchanged.


            # Store this chunk's token IDs.
            chunk_token_ids.append(token_ids)


        # -------------------------------------------------
        # 7. STORE ALL CHUNKS FOR THIS DOCUMENT
        # -------------------------------------------------

        # Add the list of tokenized chunks for this document
        # to the overall result.
        tokenized_texts.append(chunk_token_ids)


    # -----------------------------------------------------
    # 8. RETURN ALL TOKENIZED DOCUMENTS
    # -----------------------------------------------------

    return tokenized_texts


# ---------------------------------------------------------
# 9. DEFINE A FUNCTION FOR SPLITTING LONG TEXT
# ---------------------------------------------------------

# This function attempts to break one large document
# into smaller chunks.
#
# It takes:
#
# input_text:
#     One complete cleaned document.
#
# max_length:
#     Maximum permitted size for each chunk.
def split_text(input_text, max_length):


    # -----------------------------------------------------
    # 10. SPLIT THE DOCUMENT INTO WORDS
    # -----------------------------------------------------

    # str.split() without arguments splits on whitespace.
    #
    # Example:
    #
    #     "The Battle of Gettysburg"
    #
    # becomes:
    #
    #     [
    #         "The",
    #         "Battle",
    #         "of",
    #         "Gettysburg"
    #     ]
    words = input_text.split()


    # This list will eventually hold the completed chunks.
    chunks = []


    # Start with an empty current chunk.
    current_chunk = ""


    # -----------------------------------------------------
    # 11. BUILD EACH CHUNK ONE WORD AT A TIME
    # -----------------------------------------------------

    for word in words:


        # Take the current chunk and tentatively add:
        #
        #     current word
        #     +
        #     trailing space
        #
        # Example:
        #
        # current_chunk:
        #
        #     "The Battle "
        #
        # word:
        #
        #     "of"
        #
        # updated_chunk:
        #
        #     "The Battle of "
        updated_chunk = current_chunk + word + " "


        # -------------------------------------------------
        # 12. CHECK WHETHER THE CHUNK IS TOO LARGE
        # -------------------------------------------------

        # len(updated_chunk) measures the number of
        # CHARACTERS in the Python string.
        #
        # For example:
        #
        #     len("hello")
        #
        # is:
        #
        #     5
        #
        #
        # If the new word would make the chunk larger
        # than max_length, store the old chunk and begin
        # a new one.
        if len(updated_chunk) > max_length:


            # Remove leading/trailing whitespace before
            # saving the finished chunk.
            chunks.append(current_chunk.strip())


            # Start the next chunk with the word that
            # did not fit in the previous chunk.
            current_chunk = word + " "


        else:

            # The new word fits, so keep building
            # the current chunk.
            current_chunk = updated_chunk


    # -----------------------------------------------------
    # 13. SAVE THE FINAL PARTIAL CHUNK
    # -----------------------------------------------------

    # At the end of the loop there will usually still be
    # some text inside current_chunk.
    #
    # Add it to the chunks list.
    if current_chunk:
        chunks.append(current_chunk.strip())


    # Return all generated chunks.
    return chunks


# ---------------------------------------------------------
# 14. LOAD THE BERT TOKENIZER
# ---------------------------------------------------------

# Download/load the tokenizer for:
#
#     bert-base-uncased
#
# "base":
#     BERT Base architecture rather than BERT Large.
#
# "uncased":
#     Text is normalized so capitalization differences
#     are not preserved in the same way as a cased model.
#
# For example:
#
#     "Battle"
#
# and:
#
#     "battle"
#
# are effectively handled in a lowercased representation.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# ---------------------------------------------------------
# 15. DEFINE THE INPUT AND OUTPUT FILES
# ---------------------------------------------------------

# Input produced by your previous preprocessing stage.
input_file = "text_preprocessed.json"


# Output produced by this tokenization stage.
output_file = "text_tokenized.json"


# BERT's maximum input sequence is normally 512 tokens.
max_length = 512


# ---------------------------------------------------------
# 16. START FILE ERROR HANDLING
# ---------------------------------------------------------

try:


    # -----------------------------------------------------
    # 17. READ THE PREPROCESSED JSON
    # -----------------------------------------------------

    # Open the cleaned JSON data.
    with open(input_file, "r", encoding="utf-8") as f:


        # Convert JSON into Python objects.
        #
        # Conceptually:
        #
        # JSON file
        #     ↓
        # json.load()
        #     ↓
        # Python list/dictionaries
        data = json.load(f)


    # -----------------------------------------------------
    # 18. CHUNK AND TOKENIZE THE TEXT
    # -----------------------------------------------------

    # Pass:
    #
    #     BERT tokenizer
    #     loaded JSON data
    #     maximum length of 512
    #
    # into our function.
    tokenized_texts = tokenize_text(
        tokenizer,
        data,
        max_length
    )


    # -----------------------------------------------------
    # 19. WRITE TOKEN IDs TO JSON
    # -----------------------------------------------------

    with open(output_file, "w", encoding="utf-8") as f:


        # Serialize the nested Python lists containing
        # token IDs into JSON.
        json.dump(
            tokenized_texts,
            f,
            ensure_ascii=False
        )


# ---------------------------------------------------------
# 20. HANDLE FILE I/O ERRORS
# ---------------------------------------------------------

# Catch errors such as:
#
#     input file not found
#     permission denied
#     output file cannot be created
#
# This does not catch every possible error, such as
# malformed JSON.
except IOError as e:
    print(f"An error occurred while processing the file: {e}")