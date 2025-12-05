Aadarsh notes:

## Logs
- 1.12.25 - Adapted base VLM inspired from Llava Pythia. Uses Siglip/clip vision encoder and Llava meta.
- 2.12.25 - Adapted policy heads- diffusion policy and follow standard DETR architecture (ACT style). DETR output as hidden state to diffusion policy head. Pythia is underlying LM and Llava is training recipe.

## TODO
- Understand training and inference pipeline clearly.



## NOTES
This is a step in my implementation for data-efficient VLA research.

I start by adapting code from the TinyVLA paper. (https://arxiv.org/abs/2409.12514). This is because of their focus on efficient and fast models and their simple implementation of a small base VLM (pythia - https://github.com/EleutherAI/pythia adapted to Llava - https://github.com/haotian-liu/LLaVA framework) with a diffusion policy head for better dexterous output. This makes the base for a std VLA setup I want to start with.

Now, SmolVLA (https://huggingface.co/blog/smolvla ) by HuggingFace managed to train a 450M parameter model to do a decent enough job. They have a lot of thier own optimization implementations and I want to try some of my own with this object centric framework Im learning about.


Thus, I will use only a base VLM of 400M params trained by TinyVLA.(https://huggingface.co/lesjie/Llava-Pythia-400M)

Now their paper claims this model failed even in language comprehension since it is a very small VLM and showed 1.3B is better atleast in understanding. However, lets just start with this for now.

For policy head I am using diffusion head- 73M params.