text1 = """
There is nothing so intoxicating to the scientific mind as the
weird and unfamiliar. The fundamental basis of scientific thought is that an
observed truth that undermines one's understanding is yet the truth. If the
observation is not flawed, one's previous understanding must be. To the open
mind, this is not a crisis; it's an opportunity to form a new, more perfect
understanding of the world. So would it be abandoning science for a belief in
magic? Not necessarily. Rather, you would include magic in your understanding of
the physical phenomena that shape our world. Science is a path to knowledge -
one that must include and explain every observable fact, embracing all and
rejecting none. This applies to any endeavor where scientific thinking is important,
which most certainly applies to religious and historical studies. (Scientific thinking
is a type of knowledge seeking involving intentional information seeking, including
asking questions, testing hypotheses, making observations, recognizing patterns,
and making inferences.)
"""
text2 = """
Ultimate causes are something a lot of people are
concerned with to an extent. This is an atavistic trait acquired long ago for
surviving in the physical world in which there are actually causes and effects - say,
proximity to lions and being eaten. We're built to look for causal relations
between things and to be deeply satisfied when we discover a rule with cascading
implications. We're also built to be impatient with the opposite - forests of facts
from which we can't seem to extract any meaning. No matter how much people
pride themselves on logic or intellect, if their desire to believe something is strong
enough, their minds will happily weave a fiction around those wishes until those
wishes become stubborn beliefs. Thus does an opinion transmute into a putative
fact. This process, if adhered to, often leads to compromising the discernment,
judgment, and caution mentioned earlier. It can allow us to see patterns that
aren't there while also missing patterns that clearly are there.
"""

import pandas as pd
from transformers import pipeline

classifier = pipeline("text-classification")
outputs = classifier(text2)

df = pd.DataFrame(outputs)
print(df)