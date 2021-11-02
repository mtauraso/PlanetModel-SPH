using Unity.Entities;
using Unity.Physics.Systems;
using Unity.Mathematics;
using Unity.Burst;
using Unity.Jobs;
using Unity.Transforms;
using Unity.Collections;



// Update gravity field particle-by-particle
// We are going to do this in a naive fashion for now
// - Go over every pair of SPH particles O(n^2)
//   (Note: The NlogN way to do this involves tracking grav moments aligned with broadphase tree)
// - Calculate distance in this function
//   (Note: Could memoize the distances calculated in kernel function and use them here.
//          Right now we duplicate math, but also don't depend on kernel system)
// - Use a uniform-density model for the particles, with their smaller smoothing length "h" as the characteristic particle size a
//   Our force law is Equation 8 from "Softening in N-body Simulations of CollisionLes systems" (1993, Dyer & Ip)
// - Note: This choice of model means we don't use the output of the density system
//
//   Force law for r >= a:
//   |F_grav| = (G*M*M)/(r^2)
//
//   Force law for r <= a:
//   x = r/a
//   |F_grav| = ((G*M*M*x)/(a^2))*(8 - 9x + 2x^3)
//
// - Store the gravity field vector g, and the gravitational potential phi
// - Some assumptions we use now:
//    G = 1
//    F_grav(vec) = m * g(vector)
//    - Gradient(phi) = g(vector)
// 
[BurstCompile]
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(StepPhysicsWorld))]
public class GravityFieldSystem : SystemBase
{
    protected override void OnUpdate()
    {
        // Entity query that will give us only entities which have all three
        // we will eventually need ParticleSmoothing for variable smoothing lengths, but do not use it now.
        var gravParticleQuery = GetEntityQuery(typeof(ParticleMass), typeof(ParticleSmoothing), typeof(Translation));

        var entityDataLocal = gravParticleQuery.ToEntityArray(Allocator.TempJob);
        var massData = GetComponentDataFromEntity<ParticleMass>(true);
        var translationData = GetComponentDataFromEntity<Translation>(true);

        Entities.WithReadOnly(massData).WithReadOnly(translationData)
            .WithReadOnly(entityDataLocal).WithDisposeOnCompletion(entityDataLocal)
            .ForEach((Entity i, ref GravityField grav_i, // Should be write-only?
                in ParticleSmoothing smoothing_i, in Translation translation_i ) => 
            {
                float4 gravity = float4.zero;
                float3 r_i = translation_i.Value;

                // We're going to assume constant smoothing length for now
                // TODO: Variable smoothing lengths. Options:
                //       - Arbitrary symmetrization
                //       - Dyer & Ip Uniform Density Sphere formula
                //       - Switching to Price/Monaghan Kernel softening method
                float a = smoothing_i.h;

                foreach (Entity particle_j in entityDataLocal) 
                {   
                    // Skip self-contribution
                    if (particle_j == i) 
                    {
                        // Todo: Add in self-contribution for potential?
                        continue; 
                    }

                    float3 r_j = translationData[particle_j].Value;
                    float m = massData[particle_j].Value;

                    float3 displacement = r_i - r_j;                    
                    float r = math.length(displacement);
                    float grav_mag_over_r;
                    float grav_potential;

                    if (r < a) {
                        float x = r / a;
                        float x_sq = x * x;
                        float x_cube = x_sq * x;
                        float x_5th = x_sq * x_cube;
                        grav_mag_over_r = (m / (a * a * a)) * (8.0f - 9.0f * x + 2.0f * x_cube);
                        grav_potential = -(m / a) *(2.4f - 4.0f * x_sq + 3.0f * x_cube - 0.4f*x_5th) ;
                    } else {
                        grav_mag_over_r = m / (r * r * r);
                        grav_potential = -(m / r);
                    }

                    float3 grav_field = displacement * ( - grav_mag_over_r );
                    gravity += new float4(grav_field, grav_potential);
                }

                grav_i.Value = gravity;
            }).ScheduleParallel();
    }
}