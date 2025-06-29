COSTAR ="""CONTEXT:
I am currently completing a downstream data processing task for natural language processing. The sentence pairs provided in my context are the materials I prepared. I need to convert a batch of <|toxic_sentence|> into <|neutral_sentence|> versions according to the file requirements. The content in the context is some conversion examples.

OBJECTIV:
I will provide you with a batch of <|toxic_sentence|>. Please give me the <|neutral_sentence|> version to complete the task. The two languages ​​must be the same.

STYLE:
Your style should be like a rigorous programmer who understands literature, strictly completing the task without changing the original meaning and emotional expression.

TONE:
Professional.

AUDIENCE:
Adult judges who understand language will judge whether your converted <|neutral_sentence|> meets the requirements.

RESPONSE:
The returned content is toxic_sentence, neutral_sentence, lang.
The format is
{{
"toxic_sentence":"",
"neutral_sentence":"",
"lang":""
}}
Where lang is the language type: en,ru,uk,de,es,am,zh,ar,hi,it,fr,he,hin,tt,ja
"""

RISES="""
Role:
Assume you are an expert in language processing.

Inputs:
I have received a batch of toxic sentences (sentences containing harmful text) <|toxic_sentence|> and their detoxified neutral counterparts (sentences with harmful text removed) <|neutral_sentence|> in the context object.

Steps:
You can detoxify the sentences by removing harmful keywords or directly optimizing the sentences to convert them into neutral versions.

Expectation of the result:
Return the data in JSON format as follows:
{
    "toxic_sentence": "",
    "neutral_sentence": "",
    "lang": ""
}
Where:
lang is the language type: en, ru, uk, de, es, am, zh, ar, hi, it, fr, he, hin, tt, ja.
neutral_sentence is the detoxified neutral sentence, which should retain the original content while ensuring semantic similarity (measured by cosine similarity between LaBSE embeddings) and maintaining good fluency.
toxic_sentence is the original toxic sentence before detoxification.
"""
