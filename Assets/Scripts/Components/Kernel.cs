using Unity.Entities;
using Unity.Mathematics;

// TODO: how many should we keep on-chunk before we dynamically allocate?
[InternalBufferCapacity(50)]
public struct ParticleInteraction : IBufferElementData
{
    public Entity Other;
    public float4 KernelThis; // w = W(r_i-r_j,h_i) xyz = Del_i W(r_i-r_j,h_i)
    public float4 KernelSymmetric; // 1/2 (Kernel + KernelOther)
#if DEBUG_KERNEL
    public float4 KernelOther; // w = W(r_i-r_j, h_j) xyz = Del_i W(r_i-r_j,h_j)
    public Entity Self; // Only needed for intermediate queue, could be factored out
    public float distance;
#endif
}

