## DSL Related
- [Adds ability to visualise ARC-DSL](https://github.com/ArcTeamSpectral/arc-dsl) - https://arc-dsl.netlify.app/
    - Can be used to synthesise new ARC tasks I think

If reserved but unallocated memory is large try setting
26538 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management
26539 (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

expandable_segments (experimental, default: False) If set to True, this setting instructs the allocator to create CUDA allocations that can later be expanded to better handle cases where a job changing allocation sizes frequently, such as having a changing batch size. 

## Papers
- [CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay](https://openreview.net/forum?id=ac1nup9auY) 
- [CodeIt: Abstract Reasoning with Iterative Policy-Guided Program Synthesis](https://openreview.net/forum?id=JlSyXwCEIQ)
- [Intelligence at the edge of chaos](https://arxiv.org/abs/2410.02536)