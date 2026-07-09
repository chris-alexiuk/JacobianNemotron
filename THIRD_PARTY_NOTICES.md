# Third-party notices

## Anthropic Jacobian lens

The `jlens` package is based on Anthropic's public Jacobian-lens code at
commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`. It is distributed under the
Apache License 2.0 in the root `LICENSE` file and retains the upstream
copyright and SPDX headers. The local `jlens/hf.py` adaptation adds the
explicit NVIDIA Nemotron H layout (`backbone.layers`, `backbone.norm_f`, and
`backbone.embeddings`) to the Hugging Face model-layout table.

## Inputs not distributed by this repository

NVIDIA Nemotron model weights are not included. They must be obtained from the
model publisher at the pinned revision and remain subject to NVIDIA's model
license and use terms:

<https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16>

The fitted WikiText corpus is also not included. Users must review the pinned
`Salesforce/wikitext` dataset card and its underlying WikiText licensing before
materializing or redistributing corpus data. This repository's code license
does not grant rights to either the model weights or dataset content.

## Neuronpedia live lens intervention semantics

The live steering, ablation, swap, and hook-order behavior in
`nemotron_steering/interventions.py` was independently implemented against
Neuronpedia production commit
`fba06912787a1cd92fa68db2b708a7a3d1c4a5c7`, particularly
`apps/inference/neuronpedia_inference/endpoints/lens/prompt.py`.

Neuronpedia is licensed under the MIT License:

Copyright (c) 2025 Johnny Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## jlens-mood

The `nemotron_mood` package is adapted from
[`eric-tramel/jlens-mood`](https://github.com/eric-tramel/jlens-mood) at exact
commit `7b444c77c1c451068bf80c06a31aba5f4da23af7`.

`jlens-mood` is licensed under the MIT License:

Copyright (c) 2026 Eric W. Tramel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
