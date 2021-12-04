using Unity.Entities;
using Unity.Physics;
using Unity.Physics.Systems;
using Unity.Mathematics;
using Unity.Jobs;
using Unity.Burst;
using Unity.Transforms;



// Run just before Physics assembles the collision world
// Updating smoothing length for next timestep (and all the places it is cached!)
[BurstCompile]
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(ExportPhysicsWorld))]
public class ParticleSmoothingSystem : SystemBase
{
    public const float TARGET_NEIGHBORS = 50;
    protected override void OnUpdate()
    {
        Entities.ForEach((
            int entityInQueryIndex, Entity i,
            ref ParticleSmoothing smoothing_i,
            ref Scale scale_i,
            ref PhysicsCollider collider_i,
            in DynamicBuffer<ParticleInteraction> interactions) =>
        {
            // Figure out the new h
            // Iterate by averaging the current size and where we expect the
            // particle radius ought to be based on uniform particle density
            float h_prev = smoothing_i.h;
            int neighbors_prev = 0;
            foreach(var interaction in interactions)
            {
                // Only count a neighbor if its in our kernel radius
                // Symmetrized kernel or total interaction count will
                // give us contributions with increase/decrease based
                // on other particles scaling up/down.
                if (interaction.KernelThis.w > 0.0f)
                {
                    neighbors_prev += 1;
                }
            }
            smoothing_i.neighbors = neighbors_prev;

            float h_next = h_prev;
            if (neighbors_prev != 0)
            {
                float neighbors_ratio = TARGET_NEIGHBORS / (float) neighbors_prev;
                float radius_ratio = math.pow(neighbors_ratio, 1.0f / 3.0f);
                h_next = h_prev * 0.5f * (1.0f + radius_ratio);

            } else
            {
                // todo: Uh we should perobably be bigger??
                // No neighbors... do we double?
                // Ignore for the moment, need to consier when this comes up physically.
                h_next = h_prev;
            }

            // Store our new h
            // first in the ParticleSmoothing Component, which will generate the size we need for collision geometry
            smoothing_i.h = h_next;
            float particleSize = h_next * SplineKernel.Kappa();

            // Next in the collision geometry
            if (collider_i.Value.Value.Type == ColliderType.Sphere)
            {
                unsafe
                {
                    // grab the sphere pointer
                    SphereCollider* scPtr = (SphereCollider*)collider_i.ColliderPtr;
                    // DEBUG FUN
                    // Reach into particle smoothing and store the world location and radius of our sphere collider...
                    smoothing_i.sphereColliderPosRadius = new float4(scPtr->Center, scPtr->Radius);

                    // update the collider geometry
                    var sphereGeometry = scPtr->Geometry;
                    sphereGeometry.Radius = particleSize;
                    scPtr->Geometry = sphereGeometry;
                }
            }
            // Finally in the Scale Transform
            scale_i.Value *= h_next / h_prev;

        }).ScheduleParallel();
    }
}