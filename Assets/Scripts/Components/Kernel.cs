using Unity.Entities;
using Unity.Mathematics;


#if KERNEL_DYNAMIC_BUFFER
// TODO: how many should we keep before we 
//       ask for a dynamic allocation
[InternalBufferCapacity(10)]
public struct InteractionEntity : IBufferElementData
{
    public Entity e;
}

public struct InteractionKernel : IBufferElementData
{
    public float w;
}

public struct InteractionKernelGradient : IBufferElementData
{
    public float3 DelW;
}
#endif


