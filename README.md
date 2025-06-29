# Detoxification
   The project is implemented through ensemble learning, using a combination of multiple large language models to detoxify text and evaluate the detoxification effect.
## 1.结构介绍

*  The examples_data directory contains our small-scale sample content, including both officially provided and self-constructed materials.
*  The result_LLM_EnsembleLearning directory is the path to the final structure, where we separate each language to facilitate readers in finding the corresponding detoxification results.
*  The toxi_text_list directory is a summary of toxic vocabulary, including the collation of official toxic_keywords (https://huggingface.co/datasets/textdetox/multilingual_toxic_lexicon) and toxic_span (https://huggingface.co/datasets/textdetox/multilingual_toxic_spans).
*  The src_LLM_EnsembleLearning directory is a collection of source codes, including detoxification code (more_model.py) and evaluation code (evaluate.py).

## 2.Result Analysis
![img.png](img.png)
   
For details, please refer to the relevant CLEF paper, which has not been published yet (will be updated later).

## 3、 Result Analysis
The overall results are promising. For specific details, please check the performance of the Jiaozipi team on the PAN2025 official website (https://pan.webis.de/clef25/pan25-web/text-detoxification.html#data).
![img_3.png](img_3.png)
![img_4.png](img_4.png)

### Shortcomings

* The model can detoxify text in most cases, but there are still some issues, such as homophonic puns and ambiguous boundary problems. For example, in the 4th line of the Chinese results in the result set, "house" is a homophone for "好死" (hǎosǐ, meaning "a good death"), but it was detoxified as "房子" (fángzi, meaning "house"), altering the original meaning and semantics. The correct detoxification should be "这样的人没有好下场" (Such people will come to no good end).
![img_1.png](img_1.png)
####
* Another example is the 122nd line of the Chinese results, where the understanding of "neger" can have different meanings in specific contexts, and the semantics may change. It should be interpreted according to the specific context.
![img_2.png](img_2.png)
