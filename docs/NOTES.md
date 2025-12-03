Aadarsh notes:

## Logs
- 1.12.25 - Adapted base VLM inspired from Llava Pythia. Uses Siglip/clip vision encoder and Llava meta.
- 2.12.25 - Adapted policy heads- diffusion policy and follow standard DETR architecture (ACT style). DETR output as hidden state to diffusion policy head. Pythia is underlying LM and Llava is training recipe.

## TODO
- Understand training and inference pipeline clearly.