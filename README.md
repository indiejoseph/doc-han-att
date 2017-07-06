Hierarchical Attention Networks for Chinese Sentiment Classification
====================================================

This is [HAN](http://www.aclweb.org/anthology/N16-1174) version of sentiment classification, with pre-trained [character-level embedding](https://github.com/indiejoseph/chinese-char-rnn), and used [RAN](https://github.com/indiejoseph/tf-ran-cell) instead of GRU.

### Dataset
Downloaded from internet but i forget where is it ;p, the original dataset is in Simplified Chinese, i used opencc translated it into Traditional Chinese.
After 100 epochs, the valid accuracy achieved 96.31%

### Requirement
Tensorflow r1.1+

### Attention Heatmap
![attention heatmap](/attention.png)
