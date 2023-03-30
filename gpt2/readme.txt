GPT-2 text completion and compression demo
==========================================

1) Usage
--------

Extract the 117M GPT-2 model to the gpt2tc directory:

tar xtf gpt2tc-117M.tar.gz

Text completion example:

./gpt2tc g "Hello, my name is"

Use more CPU cores (only faster on server CPUs):

./gpt2tc -T 4 g "Hello, my name is"

Short Text compression and decompression example:

./gpt2tc cs "Hello, how are you ?"

./gpt2tc ds "姯敳痪"

Text compression example:

./gpt2tc c in.txt out.bin

Decompression:

./gpt2tc d out.bin out.txt

2) Using larger models
----------------------

The smallest GPT-2 model (117M) is provided in a separate
archive. Larger models can be built by downloading the TensorFlow
parameters and converting them with the attached script. Example:

# download the model to models/345M
./download_model.sh 345M

# convert it to the gpt2tc format:
python3 gpt2convert.py models/345M gpt2_345M.bin

# use it
./gpt2tc -m 345M g "Hello, how are you ?"

3) Compression results
----------------------

File          Model  Original size Compr. size  Ratio  CMIX v18
            #params        (bytes)     (bytes)  (bpb)  ratio (bpb)
book1          117M         768771      152283   1.58  1.82
book1          345M         768771      142183   1.48
book1          774M         768771      137562   1.43
book1         1558M         768771      134217   1.40

alice29.txt    117M         152089       23615   1.24  1.65
alice29.txt    345M         152089       20587   1.08
alice29.txt    774M         152089       19096   1.00
alice29.txt   1558M         152089       17382   0.91

enwik5         117M         100000       14875   1.19  1.60
enwik5         345M         100000       13511   1.08
enwik5         774M         100000       13240   1.06
enwik5        1558M         100000       12918   1.03

Notes:
- book1 comes from the Calgary corpus.
- alice29.txt comes from the Canterbury corpus.
- enwik5 contains the first 100000 bytes of the English
  Wikipedia dump of March 3, 2006
  (http://mattmahoney.net/dc/textdata.html).
- For best performance, use the UTF-8 encoding and don't mix CRLF and
  LF line breaks.
- For reference, the results of CMIX
  (http://www.byronknoll.com/cmix.html) are provided.

4) More information
-------------------

This demo has no external dependency. It is written in C and uses the
LibNC library for tensor manipulation. The CPU must support AVX2.

A similar program is used for http://textsynth.org/
