# at_bay_project_summary

installation instructions:
download https://github.com/aviadar/at_bay_project_summary/tree/master/dist/at_bay_project_summarizer-0.0.6-py3-none-any.whl and use pip to install it.
requirements.txt include all the dependencies.

usage:
create a Summarizer class (when using GPU use gpu=0)
the summarize_all function is the only one that should be used.

it's parameters include:
in_text: str - the text that should be summarized
max_token_length: int - the maximum tokens that should be after the summary process, the return value is the summarized text.
