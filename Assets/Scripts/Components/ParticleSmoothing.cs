using Unity.Entities;

// This is smoothing kernel aligned data that has to be kept on the particle
public struct ParticleSmoothing : IComponentData
{
    // Input is the particle size support domain, which is the value used in collision detection
    // to determine interactions.
    public ParticleSmoothing(float particle_size)
    {
        influenceArea = particle_size / SplineKernel.Kappa();
        supportDomain = particle_size;
        interactCount = 0;
    }
    public float h { get => influenceArea; }
    private float influenceArea; // this is the size parameter for our smoothing kernel h
    private float supportDomain; // this is the support domain kappa*H, outside of which we expect the kernel to give zero

    // How many particles are interacting with this one as given by a nonzero kernel function
    public int interactCount;

}

