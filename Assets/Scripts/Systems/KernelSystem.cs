using System;
using System.Reflection;
using UnityEngine;
using Unity.Entities;
using Unity.Physics.Systems;
using Unity.Physics;
using Unity.Jobs;
using Unity.Transforms;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Burst;


// EntityPair is a sealed type unfortunately
public struct EntityOrderedPair : IEquatable<EntityOrderedPair>
{
    private Entity EntityB;
    private Entity EntityA;

    public EntityOrderedPair(Entity A, Entity B)
    {
        EntityA = A;
        EntityB = B;
    }

    public override int GetHashCode()
    {
        return EntityA.GetHashCode() ^ EntityB.GetHashCode();
    }

    // Treats the entities as an unordered pair
    public bool Equals(EntityOrderedPair other)
    {
        return (EntityA.Equals(other.EntityA) && EntityB.Equals(other.EntityB));
    }
}


// Copied from IPhysicsSystem because this is a good idea generally
// for systems that have to run in-order repeatedly and handle their
// own data structure cleanup
interface IParticleSystem
{
    void AddInputDependency(JobHandle inputDep);

    JobHandle GetOutputDependency();
}

[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(StepPhysicsWorld))]
[AlwaysUpdateSystem]
public class KernelDataSystem : EntityCommandBufferSystem
{
    // No implmentation, this is just to have a buffer that accumulates
    // Changes to kernel dynamic buffer component
}

[BurstCompile]
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(BuildPhysicsWorld))]
[UpdateBefore(typeof(StepPhysicsWorld))]
public class KernelSystem : SystemBase, IParticleSystem
{
    private StepPhysicsWorld m_StepPhysicsWorld;
#if KERNEL_DYNAMIC_BUFFER
    private KernelDataSystem m_KernelDataSystem;
#else
    // These arrays are allocated and updated each frame.
    // If you want to use them in a system you need to:
    //
    // 1) Schedule at the right time
    //
    // [UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
    // [UpdateAfter(typeof(StepPhysicsWorld))]
    //
    // 2) Use our IParticleSystem interface. If your system creates values
    //    Needed downstream, also implement it yourself
    //
    //   Jobs that read our arrays need to be Added as an input dependency
    //   Jobs that read our arrays should wait on our output dependency,
    //   which is updated after StepPhysicsWorld runs.


    // interactionPairs has i, j pairs of interacting particle entities
    // The mapy is symmetric such that interactionPairs[i] == j <-> interactionPairs[j] == i
    public static NativeMultiHashMap<Entity, Entity> interactionPairs;

    // This is an array that maps from a pair of entities to a packed version of the kernel
    // evaluated for that pair.
    // 
    // w is the kernel function W
    // x,y,z are the derivatives of W along the spatial directions
    // With respect to the first particle in the pair.
    //
    // this means that:
    //    kernelContributions[i,j].w == kernelContributions[j,i].w;
    //    kernelContributions[i,j].xyz == kernelContributions[j,i].xyz;
#if USE_KC
    public static NativeHashMap<EntityOrderedPair, KernelContribution> kernelContributions;
#else
    public static NativeHashMap<EntityOrderedPair, float4> kernelContributions;
#endif
#endif

    public JobHandle GetOutputDependency() => outputDependency;
    private JobHandle outputDependency; // Only valid after StepPhysics runs

    // If you need to use the interaction data, register yourself so we 
    // wait for you to finish
    // TODO modify this and callers after IPhysicsJob
    public void AddInputDependency(JobHandle jh)
    {
        inputDependency = JobHandle.CombineDependencies(jh, inputDependency);
    }
    private JobHandle inputDependency;

    private int updateNumber;

    protected override void OnCreate()
    {
        var world = World.DefaultGameObjectInjectionWorld;
        m_StepPhysicsWorld = world.GetOrCreateSystem<StepPhysicsWorld>();
        m_KernelDataSystem = world.GetOrCreateSystem<KernelDataSystem>();

        inputDependency = default;
        outputDependency = default;

        updateNumber = 0;
    }

    protected override void OnDestroy()
    {
        Cleanup();
    }

    protected void Cleanup()
    {
        // Paranoid: We actually generated data, right?
        outputDependency.Complete();

        // Wait for our consumers to be done with the data before we dispose
        inputDependency.Complete();

        // Once we know we're the last ones, dump some stats to console
        if (updateNumber % 100 == 0)
        {
            var particleCount = 0;
            var interactionCount = 0;
            // Block the main thread until completion here
            Entities.ForEach((in DynamicBuffer<ParticleInteraction> a) =>
            {
                interactionCount += a.Length;
                particleCount += 1;
            }).Run();

            Debug.Log("Update #" + updateNumber.ToString());
            Debug.Log("Interaction Pairs: " + interactionCount.ToString());
            Debug.Log("Particle count: " + particleCount.ToString());
            Debug.Log("Avg interactions per particle: " + (interactionCount / particleCount).ToString());
        }

        // Go over every particle and remove all the kernel contributions.
        // On the main thread of course!
        // This is not as bad as it seems, because Clear leaves memory capacity allocated.
        Entities.ForEach((ref DynamicBuffer<ParticleInteraction> interactions) =>
        {
            interactions.Clear();
        }).Run();

        // Reset our input Dependency, so we can start collecting for the next frame
        inputDependency = default;
    }

    protected override void OnUpdate()
    {
        Cleanup();

        KernelDataSystem kernelDataSystem = m_KernelDataSystem;

        // Command buffer to store additions to DynamicBuffers for later playback
        EntityCommandBuffer ecb = kernelDataSystem.CreateCommandBuffer();

        m_StepPhysicsWorld.EnqueueCallback(SimulationCallbacks.Phase.PostCreateDispatchPairs,
           (ref ISimulation sim, ref PhysicsWorld pw, JobHandle inputDeps) =>
           { 
                int batchCount = 1; // How many interaction pairs should a worker process before returning to the pool?

                // Try Get sim.StepContext.PhasedDispatchPais with reflection
                // This is an internal array in the physics system of object pairs from the broadphase.
                // We need this array to distribute processing of pairs across threads
                // What we are doing is essentially an IBodyPairsJob, but multithreaded
                // It is too bad there is no interface for this in Physics
                var stepContextFieldInfo = sim.GetType().GetField("StepContext", BindingFlags.NonPublic | BindingFlags.Instance);
                var stepContext = stepContextFieldInfo.GetValue(sim);
                var phasedDispatchPairsFieldInfo = stepContext.GetType().GetField("PhasedDispatchPairs", BindingFlags.Public | BindingFlags.Instance);
                var phasedDispatchPairs = (NativeList<DispatchPairSequencer.DispatchPair>)phasedDispatchPairsFieldInfo.GetValue(stepContext);

                var capturePairsTest = new InteractionPairsJob()
                {
                    phasedDispatchPairs = phasedDispatchPairs,
                    bodies = pw.Bodies,
                    smoothingData = GetComponentDataFromEntity<ParticleSmoothing>(true), // RO access to particle smoothing size
                    translationData = GetComponentDataFromEntity<Translation>(true),
                    interactionDataWriter = ecb.AsParallelWriter(),
                };
                inputDeps = capturePairsTest.Schedule(phasedDispatchPairs, batchCount, inputDeps);

                m_KernelDataSystem.AddJobHandleForProducer(inputDeps);

                var disablePairs = new DisablePairsJob();
                inputDeps = disablePairs.Schedule(sim, ref pw, inputDeps);

                var scheduledJobHandle = JobHandle.CombineDependencies(inputDeps, Dependency);
               
                outputDependency = scheduledJobHandle;

                return scheduledJobHandle;
           }, Dependency);
        updateNumber++;
    }
    
    // This job nerfs future physics on any interaction pair identified in 
    // Broadphase which we will be computing. Operates in concert with InteractionPairsJob
    [BurstCompile]
    public struct DisablePairsJob : IBodyPairsJob
    {
        public void Execute(ref ModifiableBodyPair triggerEvent)
        {
            // TODO: Filter object interactions

            triggerEvent.Disable();
            // Stop physics from doing more with this collision!
            // no jacobians! No collision response!
            // only step velocity integration!
        }
    }


    [BurstCompile]
    public struct InteractionPairsJob : IJobParallelForDefer
    {
        [ReadOnly]
        public NativeList<DispatchPairSequencer.DispatchPair> phasedDispatchPairs;

        [ReadOnly]
        public NativeArray<RigidBody> bodies;

        [ReadOnly]
        public ComponentDataFromEntity<ParticleSmoothing> smoothingData;

        [ReadOnly]
        public ComponentDataFromEntity<Translation> translationData;

        [WriteOnly]
        public EntityCommandBuffer.ParallelWriter interactionDataWriter;

        public void Execute(int index)
        {
            DispatchPairSequencer.DispatchPair dispatchPair = phasedDispatchPairs[index];
            // Skip joint pairs and invalid pairs
            if (dispatchPair.IsJoint || !dispatchPair.IsValid)
            {
                return;
            }

            Entity i = bodies[dispatchPair.BodyIndexA].Entity;
            Entity j = bodies[dispatchPair.BodyIndexB].Entity;

            // TODO: Skip if both entities aren't particles..?
            //       We are disabling physics below, and should probably only do that on SPH particles

            var r_i = translationData[i].Value;
            var r_j = translationData[j].Value;

            //var distance = math.length(r_i - r_j);

            // Calculate symmetrized kernel contribution.
            // Once for derivatives w.r.t. particle i position, again for derivitives w.r.t. particle j
            // TODO: There is some duplicaiton of effort here especially on the kernel values
            // TODO: It is possible to reject pairs before performing this calculation

            // We do the version with derivative at particle i, but by symmetry
            // (I think!) the opposite order gradient is just a blanket minus sign.
            float4 kernel_ij_i = SplineKernel.KernelAndGradienti(r_i, r_j, smoothingData[i].h);
            float4 kernel_ij_j = SplineKernel.KernelAndGradienti(r_i, r_j, smoothingData[j].h);
            var kernel_ij = (kernel_ij_i + kernel_ij_j) * 0.5f;

            // By symmetry this may just be:
            // kernelji = new float4( -kernel_ij.xyz, kernel_ij.w)
            float4 kernel_ji_i = SplineKernel.KernelAndGradienti(r_j, r_i, smoothingData[i].h);
            float4 kernel_ji_j = SplineKernel.KernelAndGradienti(r_j, r_i, smoothingData[j].h);
            var kernel_ji = (kernel_ji_i + kernel_ji_j) * 0.5f;

            // Only insert to data structures later steps use for iteration if there's
            // a weight greater than zero for this pair. This saves every later calculation
            // time in iteration.
            if (kernel_ij.w > 0.0f)
            {
                // Reuse index from global order of collisions,
                // This means we can only deal with 2^30 interactions max rather than 2^31 :(
                int index_i = index << 1;
                int index_j = index_i + 1;

                // We're going to insert into the buffers of i and j a record represeting
                // The symmetrized kernel contribution of the other one
                interactionDataWriter.AppendToBuffer(index_i, i, new ParticleInteraction { Other = j, Kernel = kernel_ij });
                interactionDataWriter.AppendToBuffer(index_j, j, new ParticleInteraction { Other = i, Kernel = kernel_ji });
            }
        }
    }
}