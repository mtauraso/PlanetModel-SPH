using Unity.Entities;
using Unity.Mathematics;

public struct ParticlePressure : IComponentData
{
    public float Value;    // Pressure
}

public struct ParticlePressureGrad : IComponentData
{
    public float3 Value;    // Pressure gradient
}

