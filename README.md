# JustLM
Super easy to use library for doing LLaMA/GPT-J/MPT stuff!

## Overview
This library implements an easy to use interface to LLaMa, GPT-J and MPT, with optional Python bindings.

Context scrolling is automatic and supports a top window bar.

Additionally, "pooling" is implemented to support keeping `x` inference instances in RAM and automatically moving least recently used ones to disk, ready for retrieval.

## Documentation
Literally, just read the 2 header files in `include/`! The interface couldn't be simpler.

## Credits
Thanks to *Georgi Gerganov (ggerganov)* for having written `ggml` and `llama.cpp` C libraries, which are both extremely important parts of this project!
Also thanks to *Nomic AI* for having heavily helped me drive this project forward.
