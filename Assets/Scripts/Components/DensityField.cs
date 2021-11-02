using Unity.Entities;


public struct ParticleMass : IComponentData
{
    public float Value;    // The mass in this particle. This should not evolve dynamically
}

public struct ParticleDensity : IComponentData
{
    public float Value;   // The density at the point where this particle is located.
                          // This should only be updated by the DensityFieldSystem
}
