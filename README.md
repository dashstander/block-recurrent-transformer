# Block Recurrent Transformer

<img src="./images/brt_architecture.png" width="500px"></img>

A PyTorch implementation of [Hutchins & Schlag et al.](https://arxiv.org/abs/2203.07852v1). Owes very much to Phil Wang's [x-transformers](https://github.com/lucidrains/x-transformers). Very much in-progress.

Dockerfile, requirements.txt, _and_ environment.yaml because I love chaos.

<img src="./images/masking_pattern.png" width="500px"></img>

## Differences from the Paper (as of 2022/05/04)

* Keys and values are not shared between the "vertical" and "horizontal" directions (the standard input -> output information flow and the recurrent state flow, respectively).
* The state vectors are augmented with [Rotary Embeddings](https://blog.eleuther.ai/rotary-embeddings/) for positional encoding, instead of using learned embeddings.
* The special LSTM gate initialization is not yet implemented.
