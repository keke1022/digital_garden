---
{"dg-publish":true,"permalink":"/gpt-prompts/"}
---


## Ask for generating CL
Circumlocution is when many words are used to describe what could be said with fewer, when they have difficulty recalling a word. Pretend this is a transcript that someone is saying a speech with language disorders. If the normal speech transcript has disfluencies, keep them; otherwise do not add them. Make sure the transcriptions are realistic in daily life, and do not make exaggerations. You can make the normal speech sentence longer, but only one circumlocution is needed.

Here is an example:
a normal speech transcript: "Um I am hungry." Circumloction transcript: "Um I am feeling my stomach growling"
a normal speech transcript: "I miss my grandfather." Circumloction transcript: "I miss the father of my father"
a normal speech transcript: “It’s raining.” Circumlocution transcript: “There is water falling from the sky.”
a normal speech transcript: “It is dark.” Circumlocution transcript: “There is very little light in the area.”

These are not circumlocution speeches:
a normal speech transcript: “She’s really excited about the trip next week.” Circumlocution transcript: “She is feeling a strong sense of eagerness and anticipation about the journey that will happen next week.”
a normal speech transcript: “It’s cold in here.” Circumlocution transcript: “It’s chilly inside this place.”
a normal speech transcript: “She’s baking a cake.” Circumlocution transcript: “She’s making a sweet dessert in the oven.”

Can you generate 5 more concise normal transcripts and corresponding circumlocution transcript pairs.

## Ask to annotate
I also want to record the position of the circumlocution words. 

For example, 
Normal: It’s raining.
Circumlocution: There is water falling from the sky.

"raining" is the correct word, but the man is saying "water falling from the sky", so we mark these as 1. Because there are 7 words in the Circumlocution sentence, so the annotation is [0, 0, 1, 1, 1, 1, 1].

Keep in mind that the annotation of circumlocution is not to differ the two texts of Normal speech transcript and Circumlocution transcript. The key point is to find the word that the person forgets, and annotate these words as circumlocution. 

Can you give me some more examples?