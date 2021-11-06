//#define RECORD_ALL_COLLISIONS
//#define KERNEL_SYSTEM_EXP
// Also in KernelSystem
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Rendering;
using Unity.Transforms;
using UnityEngine;
using UnityEngine.Rendering;

// 
// Create entities following this pattern, which uses Hybrid Renderer's RenderMeshUtility
// to avoid coupling this demo too tightly to the evolving hybrid renderer.
// https://docs.unity3d.com/Packages/com.unity.rendering.hybrid@0.11/manual/runtime-entity-creation.html
//
// Because we want to delete the authoring object, the particle creation is done in the context 
// of object conversion.
//

public class ParticleAuthoring : MonoBehaviour, IConvertGameObjectToEntity
{
    // Allow setting this stuff from the editor for now
    // This is mostly to visualize the particles for debugging purpose
    // Right now everything only makes sense if the mesh is the default sphere (radius 0.5),
    // and the material is a default-enough to allow a base color override
    public Mesh mesh;
    public UnityEngine.Material material;

    // How many particles?
    public int count;

    // How big each particle (this is the initial smoothing length kh)
    public float particleRadius;

    // Radius and center of the area where we blast the particles.
    public float radius;
    public float totalMass;

    public void Convert(Entity entity, EntityManager dstManager, GameObjectConversionSystem conversionSystem)
    {
        SpawnParticles(dstManager);

        // Queue the entity corresponding to the authoring object to be removed.
        var ecb = dstManager.World.GetOrCreateSystem<EndSimulationEntityCommandBufferSystem>().CreateCommandBuffer();
        ecb.DestroyEntity(entity);
    }

    void SpawnParticles(EntityManager entityManager)
    {
        var world = entityManager.World;
        EntityCommandBuffer ecb = new EntityCommandBuffer(Allocator.TempJob);

        var desc = new RenderMeshDescription(
            mesh,
            material,
            shadowCastingMode: ShadowCastingMode.Off,
            receiveShadows: false);

        var prototype = entityManager.CreateEntity();

        RenderMeshUtility.AddComponents(
            prototype,
            entityManager,
            desc);

        // We need these in order for physics to update positions based off velocity
        // We specifically don't set Mass/Damping or anything that allows collision response
        entityManager.AddComponentData(prototype, new Translation());
        entityManager.AddComponentData(prototype, new PhysicsCollider());
#if !KERNEL_SYSTEM_EXP
        entityManager.AddComponentData(prototype, new PhysicsVelocity());
        entityManager.AddComponentData(prototype, new Rotation());
#endif

        // We need this one in order to change the color of the mesh
        entityManager.AddComponentData(prototype, new URPMaterialPropertyBaseColor());

        // So we can set custom size per-particle
        entityManager.AddComponentData(prototype, new Scale { Value = 1.0f });

        //Component data we need for SPH
        entityManager.AddComponentData(prototype, new ParticleSmoothing());
        entityManager.AddComponentData(prototype, new ParticleMass());
        entityManager.AddComponentData(prototype, new ParticleDensity());
        entityManager.AddComponentData(prototype, new GravityField());
        entityManager.AddComponentData(prototype, new ParticlePressure());
        entityManager.AddComponentData(prototype, new ParticlePressureGrad());
        entityManager.AddBuffer<ParticleInteraction>(prototype);
#if RECORD_ALL_COLLISIONS
        entityManager.AddBuffer<ParticleCollision>(prototype);
#endif

        // TODO: Explicitly run this creation on the main thread, avoid the obfuscation of a job
        //       This is being done as a parallel-for job with an entity command buffer
        //       We are probably not getting any performance benefit here because all
        //       Commands are played on the main thread, and there isn't much math here

        // TODO: Separate out parts having to do with Physical initial condition setup and particle invariant setups
        //       Physical initial conditions:
        //       - "Spherical mass distribution according to f(r)"
        //       - "Overall mass has initial velocity/angular momentum"
        //       SPH setup
        //       - "Enough small enough particles that self-gravity is modeled properly"
        //       - "Particles have a right-sh number of neighbors"
        //       Needed to run
        //       - "Particle collision geometry is the right size"
        //       - 
        var spawnJob = new SpawnParticleJob
        {
            prototype = prototype,
            Ecb = ecb.AsParallelWriter(),
            count = count,
            particleRadius = particleRadius,
            totalMass = totalMass,
            center = transform.position,
            radius = radius,
            rngFactory = world.GetExistingSystem<RandomSystem>().getRngFactory(),
        };
        var spawnHandle = spawnJob.Schedule(count, 128);
        spawnHandle.Complete();

        ecb.Playback(entityManager);
        ecb.Dispose();

        entityManager.DestroyEntity(prototype);
    } 
    

    public struct SpawnParticleJob : IJobParallelFor
    {
        // Entity we use as a prototype to stamp out many entities
        public Entity prototype;

        // How many particles
        public int count;

        // How big each particle
        public float particleRadius;

        // How much mass distributed over all particles
        public float totalMass;

        // Where to center our random particle distribution
        public float3 center;
        public float radius;  

        // Command buffer we are assembling for eventual main thread processing
        public EntityCommandBuffer.ParallelWriter Ecb;

        // How we get random numbers
        public RngFactory rngFactory;

        public void Execute(int index)
        {
            var e = Ecb.Instantiate(index, prototype);

            // Randomly generate some initial conditions
            float3 position;
            float3 particleVelocity;
            float4 baseColor;
            float particleInstanceRadius;
            using (var rng = rngFactory.GetRng())
            {
                position = center + RandomFloat3WithinSphere(rng, radius);
                //particleVelocity = RandomFloat3WithinSphere(rng, 1.0f);
                particleVelocity = float3.zero;
                particleInstanceRadius = particleRadius * (1 + rng.NextFloat(0.5f));
                baseColor = new float4(rng.NextFloat3(1.0f), 0);
            }

            // Set up a sphere collision geomtry at the appropriate center which only raises trigger events.
            // Group particles so we only get triggers when they interact with one another
            var collisionFilter = new CollisionFilter { BelongsTo = 1u << 1, CollidesWith = 1u << 1, GroupIndex = 1 };

            // TODO: Will need to alter this dynamically when we change kernel characteristic per particle.
            var physGeometry = new SphereGeometry { Center = float3.zero, Radius = particleInstanceRadius };
            var physMaterial = new Unity.Physics.Material { CollisionResponse = CollisionResponsePolicy.RaiseTriggerEvents };
            var sphereCollider = Unity.Physics.SphereCollider.Create(physGeometry, collisionFilter, physMaterial);

            // This is all the components we we need to trigger Unity.Physics to update positions of everything
            // every timestep. This is good because we can lean on Unity.Physics to keep collision lists up to date
            // This is bad because we can't control exactly how it evolves the particles every timestamp, we
            // just set PhysicsVelocity and it moves the particle for us (Yikes!)
            Ecb.SetComponent(index, e, new PhysicsCollider { Value = sphereCollider });
            Ecb.SetComponent(index, e, new Translation { Value = position });
#if !KERNEL_SYSTEM_EXP
            Ecb.SetComponent(index, e, new PhysicsVelocity { 
                Linear = particleVelocity, 
                Angular = float3.zero  // Unused by our simulation. Physics needs it
            });
            Ecb.SetComponent(index, e, new Rotation { Value = quaternion.identity }); // Unused by simulation but Physics needs it set.
            // End of Unity.Physics required components
#endif

            // This is for debug color (use later?)
            Ecb.SetComponent(index, e, new URPMaterialPropertyBaseColor { Value = baseColor });

            // For setting custom size per particle: Transform appears to only scale the mesh used to render the particle

            // This aligns the default sphere mesh with the collision geometry (should be radus kh)
            // We are multiplying by 2 because the default sphere mesh is radius 0.5
            Ecb.SetComponent(index, e, new Scale { Value = particleInstanceRadius * 2 });

            // This aligns the default sphere mesh with the size of 'h' (influence area) rather than 'kh' the support domain
            //Ecb.SetComponent(index, e, new Scale { Value = particleRadius * 2 / SplineKernel.Kappa()});

            // Set a scale factor of 1.0 for debugging
            //Ecb.SetComponent(index, e, new Scale { Value = 2.0f });



            // uniform density distribution
            var totalVolume = (4.0f * Mathf.PI / 3.0f) * radius * radius * radius;
            var density = totalMass / totalVolume;
            var particleMass = totalMass / count;

            // Set up per particle initial physical data used by SPH
            Ecb.SetComponent(index, e, new ParticleSmoothing( particleInstanceRadius ));
            Ecb.SetComponent(index, e, new ParticleMass { Value = particleMass });
            Ecb.SetComponent(index, e, new ParticleDensity { Value = density });

            // These are just data elements, we don't need to add initial values because
            // They will be calculated from what we have provided so far above
            Ecb.SetComponent(index, e, new GravityField { Value = float4.zero });
            Ecb.SetComponent(index, e, new ParticlePressure { Value = 0.0f });
            Ecb.SetComponent(index, e, new ParticlePressureGrad { Value = float3.zero });
        }

        // Uses a rejection sampling approach to generate a random float3 which is inside the sphere
        // of radius reject_radius centered at the origin. Rejection sampling approach means that when
        // interpreted as points, these should be uniformly distributed within the sphere.
        public float3 RandomFloat3WithinSphere(RngFactory.Random rng, float rejectRadius)
        {
            var rejectRadiusSq = rejectRadius * rejectRadius;
            bool candidateGood = false;
            var candidate = float3.zero;

            while (!candidateGood)
            {
                // Avoid square roots and divides.
                candidate = rng.NextFloat3(-rejectRadius, rejectRadius);
                var self_dot = candidate * candidate;
                var candidateLengthSq = self_dot.x + self_dot.y + self_dot.z;
                candidateGood = (candidateLengthSq <= rejectRadiusSq);
            }

            return candidate;
        }
    }
}
