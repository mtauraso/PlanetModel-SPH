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

#if USE_KC
public struct KernelContribution
{
    public float kernel() => Value.w;
    public float3 kernelDeriv() => Value.xyz;
    public float4 Value;
}
#endif

[BurstCompile]
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(BuildPhysicsWorld))]
[UpdateBefore(typeof(StepPhysicsWorld))]
public class KernelSystem : SystemBase, IParticleSystem

{
    private BuildPhysicsWorld m_BuildPhysicsWorld;
    private StepPhysicsWorld m_StepPhysicsWorld;

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

    private int frameNumber;

    protected override void OnCreate()
    {
        var world = World.DefaultGameObjectInjectionWorld;
        m_BuildPhysicsWorld = world.GetOrCreateSystem<BuildPhysicsWorld>();
        m_StepPhysicsWorld = world.GetOrCreateSystem<StepPhysicsWorld>();
        inputDependency = default;
        outputDependency = default;

        AllocateArrays();

        frameNumber = 0;
    }

    protected override void OnDestroy()
    {
        DisposeArrays();
    }

    protected void AllocateArrays ()
    {
        // TODO: Rework the interactionPairs and kernelContribution structures
        //       Goals: Eliminate the need to know size at physics schedule-time
        //       Ideas: 
        //          - Use a DynamicArray component on the entities
        //            Internally this looks like you set a on-chunk size, and unity goes off-chunk
        //            for storage when you expand past that size. This is ideal for a variable smoothing
        //            lengths environment (not implemented at time of writing) where you have a target
        //            number of neighbors. Could tune targets so that generally the number of interactions
        //            stays on-chunk
        //          - Push around a ref to the hashmap rather than the hash-map itself at schedule time
        //            Then perform allocate/resize operations at job time. There are examples of physics
        //            doing this with NativeList; however, not sure if it is possible/ performance-desirable with the 
        //            NativeHashmap and NativeMultiHashMap structures. Switching the multimap to a List of arrays,
        //            and the map to an array over dispach pairs looks suspiciously like a reimplementation of
        //            the underlying facilities used for DynamicArray properties.
        //       Analysis: No matter how the data is layed out, variable smoothing length will keep the number of
        //                 Interactions per particle to a set value, which will make any approach faster.
        //                 Need to build that system first and verify that it calculates correctly before optimizing
        //                 this piece. For now, set the parameter below as needed.
        //
        // Note: This is the total maximum number of interactions we hold per frame
        //       Too small -> We run out of space -> Physics crashes
        //       Too big  -> Large GC pauses each frame -> physics gets behind (Persistant alloc may fix this,
        //                   but we need to clear in DisposeArrays, which may shift
        //                   the perf issue there)
        int interactionMax = 1_000 * 100 *2; // ~ (Max particle count) * (Average interacting pairs per particle) * 2 (for i, j pair reversal)
        interactionPairs = new NativeMultiHashMap<Entity, Entity>(interactionMax, Allocator.TempJob);
#if USE_KC
        kernelContributions = new NativeHashMap<EntityOrderedPair, KernelContribution>(interactionMax, Allocator.TempJob);
#else
        kernelContributions = new NativeHashMap<EntityOrderedPair, float4>(interactionMax, Allocator.TempJob);
#endif
    }

    protected void DisposeArrays()
    {
        // Paranoid: We actually generated data, right?
        outputDependency.Complete();

        // Wait for our consumers to be done with the data before disposal
        inputDependency.Complete();

        // Once we know we're the last ones, collect some statistics
        if (frameNumber % 10 == 0)
        {
            var particleCount = 0;
            var interactionCount = 0;
            // Block the main thread until completion here
            Entities.ForEach((in ParticleSmoothing a) =>
            {
                particleCount += 1;
                interactionCount += a.interactCount;
            }).Run();
           
            Debug.Log("Frame " + frameNumber);
            Debug.Log("Interaction Pairs: " + interactionCount.ToString());
            Debug.Log("Particle count: " + particleCount.ToString());
            Debug.Log("Avg interactions per particle: " + (interactionCount/particleCount).ToString());
        }
        
        interactionPairs.Dispose();
        kernelContributions.Dispose();

        // Reset our input Dependency, so we can start collecting for the next frame
        inputDependency = default;
    }

    protected override void OnUpdate()
    {
        //Debug.Log("Kernel System OnUpdate");
        // How mass updates work:
        // This system gets an OnUpdate call every frame. For each particle it needs to 
        // sum the kernel-weighted contributions of all the particles which are "close enough" to that particle, including the original particle.
        // We're going to do this by queuing jobs that do the real work

        // First we want a list of interacting pairs
        // We will start by disposing the version from the previous frame.
        // The layout of this multimap will be that for every one way interaction where particle A can affect particle B,
        // the affected particle is the key and the affecting particle is the value
        // Because we collect these from the physics system, every A->B will also have a B->A
        DisposeArrays();
        AllocateArrays();
        

        //Debug.Log("Enqueuing Physics Callback");
        m_StepPhysicsWorld.EnqueueCallback(SimulationCallbacks.Phase.PostCreateDispatchPairs,
           (ref ISimulation sim, ref PhysicsWorld pw, JobHandle inputDeps) =>
           { 
                int batchCount = 1; // How many interaction pairs should a worker process before returning to the pool?

                // Capture local vars
                var interactionPairs = KernelSystem.interactionPairs;
                var kernelContributions = KernelSystem.kernelContributions;

                // Try Get sim.StepContext.PhasedDispatchPais with reflection
                // This is an internal array in the physics system of object pairs from the broadphase.
                // We need this array to distribute processing of pairs across threads
                // What we are doing is essentially an IBodyPairsJob, but multithreaded
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
                    interactionPairsWriter = interactionPairs.AsParallelWriter(),
                    kernelContributionWriter = kernelContributions.AsParallelWriter(),
                };
                inputDeps = capturePairsTest.Schedule(phasedDispatchPairs, batchCount, inputDeps);

                var disablePairs = new DisablePairsJob();
                inputDeps = disablePairs.Schedule(sim, ref pw, inputDeps);

                // Count our interacting neighbors
                inputDeps = Entities
                .WithReadOnly(interactionPairs)
                .ForEach((Entity i, ref ParticleSmoothing smoothing_i) =>
                {
                    smoothing_i.interactCount = 0;
                    foreach (var j in interactionPairs.GetValuesForKey(i))
                    {
                        // Count interacting neighbors which we pay calculation costs for down the line
                        smoothing_i.interactCount += 1;
                    }
                }).ScheduleParallel(inputDeps);

                var scheduledJobHandle = JobHandle.CombineDependencies(inputDeps, Dependency);
               
                outputDependency = scheduledJobHandle;

                return scheduledJobHandle;
           }, Dependency);
        
        frameNumber++;
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
        public NativeMultiHashMap<Entity, Entity>.ParallelWriter interactionPairsWriter;

        [WriteOnly]
#if USE_KC
        public NativeHashMap<EntityOrderedPair, KernelContribution>.ParallelWriter kernelContributionWriter;
#else
        public NativeHashMap<EntityOrderedPair, float4>.ParallelWriter kernelContributionWriter;
#endif

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

            // Calculate symmetrized kernel contribution
            // We do the version with derivative at particle i, but by symmetry
            // (I think!) the opposite order gradient is just a blanket minus sign.
            float4 kernel_i = SplineKernel.KernelAndGradienti(r_i, r_j, smoothingData[i].h);
            float4 kernel_j = SplineKernel.KernelAndGradienti(r_i, r_j, smoothingData[j].h);
            var kernel = (kernel_i + kernel_j) * 0.5f;

            // Only insert to data structures later steps use for iteration if there's
            // a weight greater than zero for this pair. This saves every later calculation
            // time in iteration.
            if (kernel.w > 0.0f)
            {
                // We do double updates under particle exchange because we are called once per pair
                // and both the fact of the interaction and the kernel contribution are symmetric.

                interactionPairsWriter.Add(i, j);
                interactionPairsWriter.Add(j, i);
#if USE_KC
                kernelContributionWriter.TryAdd(new EntityOrderedPair(i, j), new KernelContribution { Value = kernel });
                kernelContributionWriter.TryAdd(new EntityOrderedPair(j, i), new KernelContribution { Value = new float4(-kernel.xyz, kernel.w) });
#else
                kernelContributionWriter.TryAdd(new EntityOrderedPair(i, j), kernel);
                kernelContributionWriter.TryAdd(new EntityOrderedPair(j, i), new float4(-kernel.xyz, kernel.w));
#endif
            }
        }
    }
}