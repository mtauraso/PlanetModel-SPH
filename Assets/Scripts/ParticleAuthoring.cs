using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Transforms;
using Unity.Rendering;
using Unity.Physics.Systems;
using UnityEngine;


// Going to use this later to generate individual particles
// Has the per-particle versions of the above
struct ParticleAuthoringSettings : IComponentData
{
    // Base enitty we're copying many times
    // Flows from a code-constructed GameObject, that undergoes conversion.
    public Entity Prefab { get; set; }

    // How many particles?
    public int count;

    // TODO: Pull these from the authoring GameObject describing the whole
    // initial particle distribution (Is this a desirable property?)
    public float3 center;
    public float radius;

    // Per particle mass and velocity
    public float mass_particle;
    public float3 velocity_particle;
}

[DisallowMultipleComponent]
public class ParticleAuthoring : MonoBehaviour, IConvertGameObjectToEntity, IDeclareReferencedPrefabs
{
    // Define a uniform look for particles
    // TODO: This sort of view of particles will have to only be for debug purpose
    // TODO: Figure out how to scale/color individual particles, may require replacing this prefab
    public GameObject Prefab;

    // Only support uniform density spheres atm TODO: build other shapes
    // TODO: Float vs double vs other numerical representation?

    public int count; // How many particles are we going to spawn
    public float3 center; // Where will the center of the mass be
    public float radius; // What is the radius of the particle distribution
    public float mass_total; // What is the total mass of all particles
    public float3 velocity; // What is the total velocity of all particles
                            // TODO: Initial Angular momentum
                            // TODO: Do we need a prefab, or other material properties moving over so we can see these things?

    public void DeclareReferencedPrefabs(List<GameObject> referencedPrefabs)
    {
        Debug.Log("Called DeclareReferencedPrefabs");
        referencedPrefabs.Add(Prefab);
    }

#if false
    public void Awake()
    {
        // Note: When doing this in ECS, Start is not called before convert (or at all in my setup)
        // Only Awake is called, so we're using it. I suspect the conversion system acts and creates ECS world
        // before the traditional gameobject system would mark objects live, and allow them to begin updating.
        // But I do not fully understand why Awake must be used here.
        Debug.Log("Awake called for Particle Authoring");

        // Create a GameObject that represents a particle, including a debug-ish look for those particles
        //prototype = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        //prototype.name = "Particle Prototype";

        //Reach inside and color it green for now
        //prototype.GetComponent<Renderer>().material.SetColor("_Color", Color.green);

        // Notes: CreatePrimitive doesn't create rigid body stuff, does create a collider.
        // It could be turned off (using below), but I think I want it on for post-conversion use.
        // prefab.GetComponent<Collider>().enabled = false;
    }
#endif

    public void Convert(Entity entity, EntityManager dstManager, GameObjectConversionSystem conversionSystem)
    {
        Debug.Log("Convert called for Particle Authoring");
        // Main job here is to copy the authoring settings on to this entity so that the system below
        // can create the actual particles

        var prefab = conversionSystem.GetPrimaryEntity(this.Prefab);
        Debug.Log(prefab.ToString());

        var authoringSettings = new ParticleAuthoringSettings
        {
            Prefab = conversionSystem.GetPrimaryEntity(this.Prefab),
            count = count,
            center = center,
            radius = radius,
            mass_particle = mass_total / count,
            velocity_particle = velocity
        };
        dstManager.AddComponentData(entity, authoringSettings);
    }
}


//TODO: Look at making this a conversion system rather than a generic game system.
//TODO: Could put this all in the monobehavior per: https://docs.unity3d.com/Packages/com.unity.rendering.hybrid@0.11/manual/runtime-entity-creation.html
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateBefore(typeof(BuildPhysicsWorld))]
class SpawnParticlesSystem : SystemBase
{
    private EndSimulationEntityCommandBufferSystem m_EndSimulationEcbSystem;

    protected override void OnCreate()
    {
        base.OnCreate();
        m_EndSimulationEcbSystem = World.GetOrCreateSystem<EndSimulationEntityCommandBufferSystem>();
    }

    protected override void OnUpdate()
    {


        Entities
            .WithStructuralChanges()
            .ForEach((Entity entity, int entityInQueryIndex, ref ParticleAuthoringSettings particleAuthoringSettings) =>
        {
            var count = particleAuthoringSettings.count;
            var prefab = particleAuthoringSettings.Prefab;

            var instances = new NativeArray<Entity>(count, Allocator.Temp);
            EntityManager.Instantiate(prefab, instances);

            // Assume we're allocating in sphere and that we have a uniformly random particle distribution
            var positions = new NativeArray<float3>(count, Allocator.Temp);
            var center = particleAuthoringSettings.center;
            var radius = particleAuthoringSettings.radius;
            GenerateParticlePositions(count, center, radius, ref positions);

            // TODO actually make the particles
            for (int i = 0; i < count; i++)
            {
                if (i == 1)
                {

                }
                var instance = instances[i];
                EntityManager.SetComponentData(instance, new Translation { Value = positions[i] });
            }

            // Disappear the initial object
            EntityManager.RemoveComponent<RenderMesh>(entity);

            // Remove the settings so we only run once
            EntityManager.RemoveComponent<ParticleAuthoringSettings>(entity);
        }).WithoutBurst().Run();

    }

    // Use a rejection sampling approach to generate 3-space coordinates
    // within the radius of the position given in argument.
    // Returns results in final argument, assumes it is already allocated and of appropriate size
    private void GenerateParticlePositions(in int count, in float3 position, in float radius, ref NativeArray<float3> points, int seed = 1)
    {
        var random = new Unity.Mathematics.Random((uint)seed);

        var reject_radius_sq = radius * radius;

        for (int i = 0; i < count; i++)
        {
            var candidate = random.NextFloat3(-radius, radius);
            var self_dot = candidate * candidate;
            var candidate_length_sq = self_dot.x + self_dot.y + self_dot.z;
            if (candidate_length_sq <= reject_radius_sq)
            {
                points[i] = candidate;
            } else {
                i--; // Reject candidate and try again next loop
            }
        }
    }
}
