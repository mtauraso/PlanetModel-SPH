using Unity.Entities;
using Unity.Mathematics;


#if KERNEL_DYNAMIC_BUFFER
// TODO: how many should we keep on-chunk before we dynamically allocate?
[InternalBufferCapacity(8)]
public struct ParticleInteraction : IBufferElementData
{
    public Entity Other;
    public float4 Kernel;
}
#endif


