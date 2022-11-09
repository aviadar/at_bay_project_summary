from transformers import pipeline, AutoTokenizer


def divide_chunks(l, n=1024):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class Summarizer:
    def __init__(self, gpu=0):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", model_max_len=1024)
        if gpu == 0:
            self.summarizer = pipeline("summarization",
                                       device=0,
                                       model="facebook/bart-large-cnn",
                                       tokenizer=self.tokenizer)
        else:
            self.summarizer = pipeline("summarization",
                                       model="facebook/bart-large-cnn",
                                       tokenizer=self.tokenizer)

    def _summarize(self, in_text, max_length=130, min_length=30, do_sample=False):
        return self.summarizer(in_text, max_length=max_length,
                               min_length=min_length,
                               do_sample=do_sample,
                               truncation=True)[0]['summary_text']

    def summarize_all(self, in_text, max_token_length=1024):
        was_sum_done = False
        summarized_txt = ''

        tokenized_txt = self.tokenizer.tokenize(in_text)
        while len(tokenized_txt) > max_token_length:
            token_chunks = divide_chunks(tokenized_txt)
            summarized_txt = ''
            for token_chunk in token_chunks:
                summarized_txt += self._summarize(self.tokenizer.convert_tokens_to_string(token_chunk))
                summarized_txt += ' '

            tokenized_txt = self.tokenizer.tokenize(summarized_txt)

        if was_sum_done is False:
            summarized_txt += self._summarize(in_text)

        return summarized_txt


