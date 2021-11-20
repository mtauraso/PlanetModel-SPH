using Unity.Entities;
using Unity.Mathematics;


public struct GravityField : IComponentData
{
    // Going to try storing this as 
    // w: gravitational potential (Phi)
    // xyz: Gravitational field vector (-gradient(Phi) = g) where F_grav = g*m
    public float4 Value;

    // For the current timestep, count of number of particles that contributed
    // and number of approximations used
    public int numParticles;
    public int numApprox;

    public float3 GradPotential
    {
        get => Value.xyz;
    }
    public float Potential
    {
        get => Value.w;
    }
    public float3 FieldVector
    {
        get => -Value.xyz;
    }
}
