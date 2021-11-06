using Unity.Entities;
using Unity.Mathematics;

// This is smoothing kernel aligned data that has to be kept on the particle
public struct ParticleSmoothing : IComponentData
{
    // Input is the particle size support domain, which is the value used in collision detection
    // to determine interactions.
    public ParticleSmoothing(float particle_size)
    {
        influenceArea = particle_size / SplineKernel.Kappa();
        supportDomain = particle_size;
        sphereColliderPosRadius = float4.zero;
        neighbors = 0;
    }
    public float h { 
        get => influenceArea; 
        set 
        { 
            influenceArea = value;
            supportDomain = SplineKernel.Kappa() * value;
        }
    }
    public float ParticleSize
    {
        get => supportDomain;
    }
    public float influenceArea; // this is the size parameter for our smoothing kernel h
    public float supportDomain; // this is the support domain kappa*H, outside of which we expect the kernel to give zero
    public float4 sphereColliderPosRadius; // Updated by smoothing system, debug on sphere collider maths  xyz = center, w = radius
    public int neighbors; // Debug neighbor counts
}

