using Unity.Entities;
using Unity.Physics;
using Unity.Physics.Systems;
using Unity.Mathematics;
using Unity.Burst;
using Unity.Jobs;
using Unity.Transforms;
using Unity.Collections;
using System;
using static Unity.Physics.Broadphase;
using Unity.Jobs.LowLevel.Unsafe;

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
[UpdateAfter(typeof(BuildPhysicsWorld))]
[UpdateBefore(typeof(StepPhysicsWorld))]
public class GravityFieldSystem : SystemBase
{
    public const bool k_TreeGrav = false;

    private StepPhysicsWorld m_StepPhysicsWorld;

    protected override void OnCreate()
    {
        var world = World.DefaultGameObjectInjectionWorld;
        m_StepPhysicsWorld = world.GetOrCreateSystem<StepPhysicsWorld>();
    }

    protected override void OnUpdate()
    {
        if (k_TreeGrav) { 
            OnUpdateTree();
            OnUpdateNSquared();
        } else { 
            OnUpdateNSquared(); 
        }
    }



    public void OnUpdateNSquared()
    {
        // Entity query that will give us only entities which have all three
        // we will eventually need ParticleSmoothing for variable smoothing lengths, but do not use it now.
        var gravParticleQuery = GetEntityQuery(typeof(ParticleMass), typeof(ParticleSmoothing), typeof(Translation));

        var entityDataLocal = gravParticleQuery.ToEntityArray(Allocator.TempJob);
        var massData = GetComponentDataFromEntity<ParticleMass>(true);
        var translationData = GetComponentDataFromEntity<Translation>(true);

        float gravConstant = 1.0f;

        Entities
            .WithReadOnly(massData)
            .WithReadOnly(translationData)
            .WithReadOnly(entityDataLocal)
            .WithDisposeOnCompletion(entityDataLocal)
            .ForEach((Entity i, 
                ref GravityField grav_i, // Should be write-only?
                in ParticleSmoothing smoothing_i, 
                in Translation translation_i ) => 
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
                    } 
                    else 
                    {
                        grav_mag_over_r = m / (r * r * r);
                        grav_potential = -(m / r);
                    }

                    float3 grav_potential_gradient = displacement * grav_mag_over_r;
                    gravity += new float4(gravConstant * grav_potential_gradient, gravConstant * grav_potential);
                }

                grav_i.Value = gravity;
            }).ScheduleParallel();
    }

    public void OnUpdateTree()
    {
        m_StepPhysicsWorld.EnqueueCallback(SimulationCallbacks.Phase.PostBroadphase,
            (ref ISimulation sim, ref PhysicsWorld pw, JobHandle inputDeps) =>
            {
                // A little paranoid since we're attaching to a custom callback that only exists
                // in modified unity physics
                if (sim.Type != SimulationType.UnityPhysics)
                {
                    throw new NotImplementedException("SPH KernelSystem Only works with Unity Physics");
                }

                // Get the tree!!
                Tree BVH = pw.CollisionWorld.Broadphase.DynamicTree;
                // bodies in broadphase are m_Bodies
                // m_bodies is set up as [[dynamic bodies][Static bodies...]]
                // 
                // The dynamic tree is only over dynamic bodies, likewise for static
                // Indicies in tree are within the sub-array, not overal bodies.
                // For dynamic bodies these are the same indicies, since dynamic bodies are first
                //
                // Leaf processors receive indicies for the index space
                //
                // None of the collision/distance algorithms I'm seeing let one stop processing
                // at a non-leaf node, they are all full traversals, so we need our own

                NativeArray<GravitationalMoment> moments = new NativeArray<GravitationalMoment>(BVH.Nodes.Length, Allocator.TempJob);

                // Annotate the tree with gravity
                var GenerateMomentsSTJob = new GenerateMomentsSTJob
                {
                    // Inputs:
                    ParticleTree = BVH,
                    Bodies = pw.Bodies,
                    MassData = GetComponentDataFromEntity<ParticleMass>(true),

                    // Need a place to stick the moments
                    Moments = moments,
                };
                inputDeps = GenerateMomentsSTJob.Schedule(inputDeps);

                // Second things second, treewalk per particle
                // First version: Walk the tree to leaves, evaluate at leaves
                // Second version: Check intermediate nodes and use prior data
                var particleGravityJob = Entities
                //.WithReadOnly(BVH)
                //.WithReadOnly(moments)
                //.WithDisposeOnCompletion(moments)
                .ForEach((
                    ref GravityField gravityField, 
                    in Translation translation, 
                    in ParticleMass mass) =>
                {
                    // For this particle again do a depth-first treewalk
                    // 
                    // Push the top of BVH onto the stack
                    //
                    // Pop the stack, repeatedly until empty:
                    //
                    // If we are a leaf node:
                    //   Compute the gravity contribution of the leaf node on the particle
                    //   Add this to the running sums of grav potential and field strength
                    //   Do not need entity here, just go from BVH Rigid body index -> Grav moment
                    //      This will have the CM and mass of the particle
                    //
                    // If we are a non-leaf node:
                    //   Look up our Gravitational moment to get CM
                    //   Look up the Node AABB to get maximum extent
                    //   Pass both to acceptance function
                    //   If we accept the approximation of this node:
                    //       Use grav moment to add to particle gravity contribution
                    //
                    //   If we do not accept the approximation of this node:
                    //       Push this node's children only on to the stack for processing
                    //
                    // When we have exhausted the queue write out the particle gravity contribution
                });
                inputDeps = particleGravityJob.ScheduleParallel(inputDeps);

                return inputDeps;
            });
    }

    // A gravitational moment corresponding to mass in some bounding volume
    //
    // Stores the center of mass in world coordinates and
    // the first moment (mass for point approximation)
    // in a packed float4.
    struct GravitationalMoment
    {
        public GravitationalMoment(float3 CM, float firstMoment)
        {
            m_packedCMfirstMoment = new float4(CM, firstMoment);
        }
        public float3 CenterOfMass
        {
            get => m_packedCMfirstMoment.xyz;
        }
        public float3 MonopoleMoment
        {
            get => m_packedCMfirstMoment.w;
        }

        private float4 m_packedCMfirstMoment;
    }

    // This is going to iterate the whole tree and add up moments for the relevant node
    // For now Single Threaded:
    //     If want to multithread: Look at CollisionWorld.cs, Broadphase.cs and BoundingVolumeHierarcy.cs for examples
    //     May be possible to process queue in parallel, but it also may be slower to do it that way.
    // TODO: Look at burst optimizations in other Unity.Physics tree code
    [BurstCompile]
    struct GenerateMomentsSTJob: IJob
    {
        // Input
        [ReadOnly] public Tree ParticleTree;
        [ReadOnly] public NativeArray<RigidBody> Bodies;
        [ReadOnly] public ComponentDataFromEntity<ParticleMass> MassData;
        
        // Output
        public NativeArray<GravitationalMoment> Moments;

        public void Execute()
        {
            // Will need to walk the tree This is a depth-first traversal, doing this primarily to limit stack/queue length
            //
            // Push the root node of the tree on the stack
            //
            // If there are nodes on the stack, pop one
            // For Leaves:
            //  Lookup from rigid body index in tree -> entity -> Mass component data
            //    (can we get speedup by pre-computing this lookup?)
            //  Assemble relevant moments for this object at center-of mass for the leaf
            //  Store themm in the appropriate spot in Moments array
            //
            // For Non-Leaves:
            //  If All leaves have computed moments:
            //     Calculate our moment from children's moments
            //     Do so at a reference point that is center-of-mass to this volume
            //     Store the moment and center-of-mass in the corresponding place in Moments array
            //  If Some leaves have uncomputed moments:
            //     Push ourselves and all uncomputed leaves on the stack 
            //    
        }
    }
}