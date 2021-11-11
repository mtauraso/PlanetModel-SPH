using UnityEngine;
using Unity.Burst;
using Unity.Mathematics;
using Unity.Transforms;


[BurstCompile]
static class SplineKernel
{
    // Todo: Abstract this as an interface so caller can use generalized kernel and swap out kernel function
    //
    // Todo perf: Profile overall calculations first
    // Todo perf: Create a lookup table at startup so we can do lookup + newtons method for
    //            per-frame calculations.
    // Todo perf: Rewrite so whatever inner loop calls this has 1/h or 1/size pre-computed
    //            this is coming from the particle and won't necessarily change every frame
    //            Could also extend this to 1/(Pi*h^3), and provide a per particle numerical
    //            holding area for generalized kernel functions to use in their per-interaction
    //            calculation. Consider that fp divides are sufficiently pipelined and sometimes
    //            one microcode op, if there are enough multiply/add fp. 
    //            (C++ answer on microoptimizing fp ops: https://stackoverflow.com/questions/4125033/floating-point-division-vs-floating-point-multiplication )

    // This calculates the kernel function for smooth particle hydrodynamics
    //
    // We're using the spline kernel of Monaghan and Lattanzio (1983), which I've copied from
    // Numerical Methods in Astrophysics An Introduction by Bodenheimer et. al. (2007) p 122
    //
    // Interface:
    //      float distance The euclidean distance between the relevant displacement vectors
    //                     Note: We are currently only doing spherically symmetric and even kernel functions
    //      float h        The characteristic size, see below:
    //
    // Current requirements of a kernel function W(x-x',h):
    //   1) Normalization condition: Area under the kernel function evaluated across all space is 1
    //      Int(W(x-x', h),dx') = 1
    //   2) Delta function property: as the smoothing length h approaches zero, W approaches the delta function
    //      This is based on h and defines the influence range
    //      lim(W(x-x', h), h, 0) = delta(x-x')
    //   3) Compactness property: For sufficiently distant points W is zero
    //      This defines the support domain of the kernel function
    //      W(x-x',h) = 0 when abs(x-x') > kh
    //   4) W is an even function W(-x,h) = W(x, h)
    //      Todo: try to keep this assumption inside the kernel function implementation


    // It should always be true that Kernel(Kappa()*h, h) = 0.0f because of compactness
    // TODO: learn to write tests in unity!
    static public float Kappa() => 2.0f;

    // Return true if the particles could interact at either of the given smoothing lengths
    static public bool Interacts(float3 r_i, float3 r_j, float size_i, float size_j)
    {
        float3 displacement = (r_i - r_j);
        float distanceSq = math.dot(displacement, displacement);
        return 
            (distanceSq < size_i * size_i * Kappa() * Kappa()) || 
            (distanceSq < size_j * size_j * Kappa() * Kappa());
    }

    static public float Kernel(float distance, float size)
    {
        // Early return without doing math if we're outside range
        // This should not happen very often if our collision triggering is tuned properly
        // TODO: Add a warning log here?
        float retval = 0;
        if (distance >= size * Kappa())
        {
            return retval;
        }

        // The exposition of the kernel uses a size where W -> 0 as distance -> 2h
        // Therefore kappa is 2.

        var r_over_h = distance / size;
        var pi_h_cube = Mathf.PI * size * size * size;
        // So far we have 1 divide, 3 multiplies

        if (distance < size)
        {
            float r_over_h_sq = r_over_h * r_over_h;
            float numerator = 1.0f - 1.5f * r_over_h_sq + 0.75f * (r_over_h_sq * r_over_h);
            retval = numerator / pi_h_cube;
            // This block adds 1 divide, 4 multiplies, 2 adds
        }
        else
        {
            float inner_term = 2.0f - r_over_h;
            float numerator = inner_term * inner_term * inner_term;
            float denominator = 4.0f * pi_h_cube;
            retval = numerator / denominator;
            // This block adds 1 divide, 3 multiplies, 1 add
        }
        return retval;
    }

    // Kernel gradient w.r.t. to particle i
    // Pass in displacements of particles i and j because
    // We need to get the displacement vector direction correct here.
    static public float3 KernelGradienti(float3 r_i, float3 r_j, float size)
    {
        float3 displacement = (r_i - r_j);
        float distance = math.length(displacement);

        return displacement * ( KernelDeriv(distance, size) / distance );
    }

    static public float4 KernelAndGradienti(float3 r_i, float3 r_j, float size)
    {
        float3 displacement = (r_i - r_j);
        float distance = math.length(displacement);

        float3 kernel_deriv = displacement * ( KernelDeriv(distance, size) / distance);
        float kernel = Kernel(distance, size);

        return new float4(kernel_deriv, kernel);
    }

    // Derivative of the kernel with respect to distance argument.
    // Useful for getting gradient in this kernel where there is spherical symmetry
    static public float KernelDeriv(float distance, float size)
    {
        // Early return without doing math if we're outside range
        // This should not happen very often if our collision triggering is tuned properly
        // TODO: Add a warning log here, or a counter?
        float retval = 0;
        if (distance >= size * Kappa())
        {
            return retval;
        }

        // The exposition of the kernel uses a size where W -> 0 as distance -> 2h

        var r_over_h = distance / size;
        var pi_h_4th = Mathf.PI * size * size * size * size;
        // So far we have 1 divide, 4 multiplies

        if (distance < size)
        {
            float r_over_h_sq = r_over_h * r_over_h;
            float numerator = 3.0f * r_over_h + 2.25f * r_over_h_sq;
            retval = numerator / pi_h_4th;
            // This block adds 1 divide, 3 multiplies, 1 add
        }
        else
        {
            float inner_term = 2.0f - r_over_h;
            float numerator = -3.0f * inner_term * inner_term;
            float denominator = 4.0f * pi_h_4th;
            retval = numerator / denominator;
            // This block adds 1 divide, 3 multiplies, 1 add
        }
        return retval;
    }
}