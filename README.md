# Text-Summarization
Automated text summarization using NLP techniques &amp; Deep Learning.

## Extractive Summary
The extractive summarizer uses sentences from the input dataset itself to generate a summary.
This is done by ranking the sentences in order of their importance using a text ranking implementation.
The top ranked sentences are then arranged in order of their appearance in the original text.
This summary is then written to a summary prefixed text file.

The GloVe embeddings used in the program can be downloaded from [here](http://nlp.stanford.edu/data/glove.6B.zip).

### Usage
The default params limit the summary to be five sentences long. Use the ```
--length``` parameter to customize.

```commandline
python --file sample_text.txt --embeddings glove.6B.100d.txt --length 5
```

If the specified summary length is more than the length of the input text, 
the entire input text is written as the summary.
