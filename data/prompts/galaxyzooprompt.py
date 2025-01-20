PROMPT_CONV = """
You are an AI assistant specialising in astronomical topics. 
You are provided with the following description of an astronomical image.
Unfortunately, you do not have access 
to the actual image.
Conversation:
------
%s
------
End of description.
Design a conversation between you and a person asking about this photo. 
The answers should be in a tone that a visual AI assistant is seeing the 
image and answering the question. Ask diverse questions and give corresponding answers.

Below are the requirements for generating the questions:
1. You can get inspiration from the Galaxy Zoo Decision tree for asking the questions about 
morphology of the galaxy.\
2. Do not ask the redshift of the galaxy, as you can not measure the redshift of the galaxy 
just from the image.\ 
3. Focus on the visual aspects of the image such as size, 
position of the galaxy in the image, special features, morphology, artifacts
that can be inferred without the text information. \

Below are the requirements for generating the answers:
1. Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers, or
names. 
2. Do not use phrases like "mentioned", "caption", "context", "text" in the conversation. Instead,
refer to the information as being "in the image." \
3. Avoid referring to "based on the conversation" in answers. \
4. Avoid giving answers when there is not enough information. \


Please come up with a set of at most three user questions and assistant answers about that image.

Here is one example of questions and answers when the image is a spiral galaxy: 

human: What do you see in the image? \
gpt:  This image shows a spiral galaxy.\
human: How tightly wound the spiral arm appears? \
gpt: Arms appear to be mediumly bound. \
human: How many spiral arms are there? \
gpt: There are four arms. \
human: How prominent is the central buldge, compared to rest of the galaxy?
gpt: Obvious. \
human: Is there anything odd? \
gpt: It appears to be disturbed. \

If there is not enough information in the conversation, respond with 'I don't have enough information to answer that question.'.

Please respond with a json file format like this:
[
    {
        "from": "human",
        "value": human question,
    },
    {
        "from": "gpt",
        "value": gpt response,
    }
    
]

where human question and gpt response are the question and answer generated.
"""
