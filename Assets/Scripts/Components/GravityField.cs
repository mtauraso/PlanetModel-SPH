using Unity.Entities;
using Unity.Mathematics;


public struct GravityField : IComponentData
{
    // Going to try storing this as 
    // w: gravitational potential (Phi)
    // xyz: Gravitational field vector (-gradient(Phi) = g) where F_grav = g*m
    public float4 Value;
}
