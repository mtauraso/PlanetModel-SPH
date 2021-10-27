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


// Copied from IPhysicSystem because its a good idea generally
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
        if (frameNumber % 100 == 0)
        {
            var uniqueKeys = interactionPairs.GetUniqueKeyArray(Allocator.Temp);
            float avg = 0.0f;
            int count = 0;
            foreach(var key in uniqueKeys.Item1)
            {
                var n = interactionPairs.CountValuesForKey(key);
                avg = (avg * count + n) / (count + 1.0f);
                count++;
            }
            Debug.Log("Frame " + frameNumber);
            Debug.Log("Interaction Pairs: " + (interactionPairs.Count()/2.0f).ToString());
            Debug.Log("Particle count: " + (count / 2.0f).ToString());
            Debug.Log("Avg interactions per particle: " + avg.ToString());
            uniqueKeys.Item1.Dispose();
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
               bool new_method = true;

               if (new_method)
               {
                   // Try Get sim.StepContext.PhasedDispatchPais with reflection
                   // This is an internal array in the physics system of object pairs from the broadphase.
                   // We need this array to distribute processing of pairs across threads
                   // What we are doing is essentially an IBodyPairsJob, but multithreaded
                   var stepContextFieldInfo = sim.GetType().GetField("StepContext", BindingFlags.NonPublic | BindingFlags.Instance);
                   var stepContext = stepContextFieldInfo.GetValue(sim);
                   var phasedDispatchPairsFieldInfo = stepContext.GetType().GetField("PhasedDispatchPairs", BindingFlags.Public | BindingFlags.Instance);
                   var phasedDispatchPairs = (NativeList<DispatchPairSequencer.DispatchPair>)phasedDispatchPairsFieldInfo.GetValue(stepContext);

                   var capturePairsTest = new InteractionPairsJob3()
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
               }
               else
               {
                   var capturePairs = new InteractionPairsJob2()
                   {
                       interactionPairsWriter = interactionPairs.AsParallelWriter(),
                       kernelContributionWriter = kernelContributions.AsParallelWriter(),
                       smoothingData = GetComponentDataFromEntity<ParticleSmoothing>(true), // RO access to particle smoothing size
                       translationData = GetComponentDataFromEntity<Translation>(true)
                   };
                   inputDeps = capturePairs.Schedule(sim, ref pw, inputDeps);
               }
               var scheduledJobHandle = JobHandle.CombineDependencies(inputDeps, Dependency);
               
               outputDependency = scheduledJobHandle;

               return scheduledJobHandle;
           }, Dependency);

#if false// Old way to do this with a physics ITriggerEvents job
        m_StepPhysicsWorld.EnqueueCallback(SimulationCallbacks.Phase.PostCreateDispatchPairs,
           (ref ISimulation sim, ref PhysicsWorld pw, JobHandle inputDeps) =>
           {
               var capturePairs = new InteractionPairsJob2()
               {
                   interactionPairsWriter = interactionPairs.AsParallelWriter(),
                   kernelContributionWriter = kernelContributions.AsParallelWriter(),
                   smoothingData = GetComponentDataFromEntity<ParticleSmoothing>(true), // RO access to particle smoothing size
                   translationData = GetComponentDataFromEntity<Translation>(true),
                   frameNumber = frameNumber,
               };

               this.interactionsJobHandle = capturePairs.Schedule(sim, ref pw, inputDeps);

               return interactionsJobHandle;
           }, 
           Dependency);
 
        var translationData = GetComponentDataFromEntity<Translation>(true); // RO access to particle positions
        var capturePairs = new InteractionPairsJob()
        {
            interactionPairsWriter = interactionPairs.AsParallelWriter(),
            kernelContributionWriter = kernelContributions.AsParallelWriter(),
            smoothingData = GetComponentDataFromEntity<ParticleSmoothing>(true), // RO access to particle smoothing size
            translationData = translationData,
            frameNumber = frameNumber,
        };

        // Schedule a collision job to go through all interacting i, j pairs, compute kernel + derivatives, and put them in the hashmap
        Dependency = capturePairs.Schedule(m_StepPhysicsWorld.Simulation, ref m_BuildPhysicsWorld.PhysicsWorld, Dependency);

        interactionsJobHandle = Dependency;
#endif
        
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
    public struct InteractionPairsJob3 : IJobParallelForDefer
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

            interactionPairsWriter.Add(i, j);
            interactionPairsWriter.Add(j, i);

            var r_i = translationData[i].Value;
            var r_j = translationData[j].Value;

            //var distance = math.length(r_i - r_j);

            // Calculate symmetrized kernel contribution
            // We do the version with derivative at particle i, but by symmetry
            // (I think!) the opposite order gradient is just a blanket minus sign.
            float4 kernel_i = SplineKernel.KernelAndGradienti(r_i, r_j, smoothingData[i].h);
            float4 kernel_j = SplineKernel.KernelAndGradienti(r_i, r_j, smoothingData[j].h);
            var kernel = (kernel_i + kernel_j) * 0.5f;

            // We do double updates under particle exchange because we are called once per pair
            // and both the fact of the interaction and the kernel contribution are symmetric.
#if USE_KC
            kernelContributionWriter.TryAdd(new EntityOrderedPair(i, j), new KernelContribution { Value = kernel });
            kernelContributionWriter.TryAdd(new EntityOrderedPair(j, i), new KernelContribution { Value = new float4(-kernel.xyz, kernel.w) });
#else
            kernelContributionWriter.TryAdd(new EntityOrderedPair(i, j), kernel);
            kernelContributionWriter.TryAdd(new EntityOrderedPair(j, i), new float4(-kernel.xyz, kernel.w));
#endif

            /*
            if (index % 10000 == 0)
            {
                Debug.Log("Got here index " + index.ToString());
            }
            */
        }
    }

    // TODO: Perf! Parallelize this! Requires some looking into physics internals, removing reliance on being an IBodyPairsJob
    //       This means we're going to have to be an IParallel job (similar to ParallelCreateContactsJob in NarroPhase)
    //       and hope that we have access to the right parts of physics. We will probably need to stream object pairs using NativeStream
    //       similar to how ParalllelCreateContactsJob works.
    //       Every frame is blocked for ~4ms (xcxc Update!) on this, because all of SPH depends on interaction pairs 
    [BurstCompile]
    public struct InteractionPairsJob2 : IBodyPairsJob
    {
        // Note that the only job this has is to record adjacency from one entity to another
        // Under the hood an entity is "just" two integers. There are probably some space trade-offs that
        // could be made here. Thematically this is a very large (#particles x #particles) sparse binary array
        // which we are storing as a multi-map.
        
        [WriteOnly]
        public NativeMultiHashMap<Entity, Entity>.ParallelWriter interactionPairsWriter;

        [WriteOnly]
#if USE_KC
        public NativeHashMap<EntityOrderedPair, KernelContribution>.ParallelWriter kernelContributionWriter;
#else
        public NativeHashMap<EntityOrderedPair, float4>.ParallelWriter kernelContributionWriter;

#endif

        [ReadOnly]
        public ComponentDataFromEntity<ParticleSmoothing> smoothingData;

        [ReadOnly]
        public ComponentDataFromEntity<Translation> translationData;

        //[ReadOnly]
        //public int frameNumber;


        public void Execute(ref ModifiableBodyPair triggerEvent)
        {
            triggerEvent.Disable();
            // Stop physics from doing more with this collision!
            // no jacobians! No collision response!
            // only step velocity integration!


            // Note: This will add some things that are not SPH particles until we have a collision filter
            //       The math part is written so it will just crash if we are passed not-a-particle
            var i = triggerEvent.EntityA;
            var j = triggerEvent.EntityB;

            //interactionPairsWriter.Add(i, j);
            //interactionPairsWriter.Add(j, i);
            
            var r_i = translationData[i].Value;
            var r_j = translationData[j].Value;
            
            //var distance = math.length(r_i - r_j);

            // Calculate symmetrized kernel contribution
            // We do the version with derivative at particle i, but by symmetry
            // (I think!) the opposite order gradient is just a blanket minus sign.
            float4 kernel_i = SplineKernel.KernelAndGradienti(r_i, r_j, smoothingData[i].h);
            float4 kernel_j = SplineKernel.KernelAndGradienti(r_i, r_j, smoothingData[j].h);
            var kernel = (kernel_i + kernel_j) * 0.5f;

            // We do double updates under particle exchange because we are called once per pair
            // and both the fact of the interaction and the kernel contribution are symmetric.
#if USE_KC
            kernelContributionWriter.TryAdd(new EntityOrderedPair(i, j), new KernelContribution { Value = kernel });
            kernelContributionWriter.TryAdd(new EntityOrderedPair(j, i), new KernelContribution { Value = new float4(-kernel.xyz, kernel.w) });
#else
            kernelContributionWriter.TryAdd(new EntityOrderedPair(i, j), kernel);
            kernelContributionWriter.TryAdd(new EntityOrderedPair(j, i), new float4(-kernel.xyz, kernel.w));
#endif
        }
    }

#if false // old ITriggerEventsJob way
    // TODO Name this better?
    // 
    // This gets called once per frame on every pair of particles that currently have intersecting
    // collision geometry. This means they are within particle_radius of each other.
    //
    [BurstCompile]
    public struct InteractionPairsJob : ITriggerEventsJob
    {
        // Note that the only job this has is to record adjacency from one entity to another
        // Under the hood an entity is "just" two integers. There are probably some space trade-offs that
        // could be made here. Thematically this is a very large (#particles x #particles) sparse binary array
        // which we are storing as a multi-map.
        [WriteOnly]
        public NativeMultiHashMap<Entity, Entity>.ParallelWriter interactionPairsWriter;

        [WriteOnly]
        public NativeHashMap<EntityOrderedPair, float4>.ParallelWriter kernelContributionWriter;

        [ReadOnly]
        public ComponentDataFromEntity<ParticleSmoothing> smoothingData;

        [ReadOnly]
        public ComponentDataFromEntity<Translation> translationData;

        [ReadOnly]
        public int frameNumber;
        public void Execute(TriggerEvent triggerEvent)
        {
            // Note: This will add some things that are not SPH particles until we have a collision filter
            //       The math part is written so it will just crash if we are passed not-a-particle
            var i = triggerEvent.EntityA;
            var j = triggerEvent.EntityB;

            var r_i = translationData[i].Value;
            var r_j = translationData[j].Value;
            // Calculate distance
            var distance = math.length(translationData[i].Value - translationData[j].Value);

            // Calculate symmetrized kernel contribution
            // We do the version with derivative at particle i, but by symmetry
            // (I think!) the opposite order gradient is just a blanket minus sign.
            float4 kernel_i = SplineKernel.KernelAndGradienti(r_i, r_j, smoothingData[i].h);
            float4 kernel_j = SplineKernel.KernelAndGradienti(r_i, r_j, smoothingData[j].h);
            var kernel = (kernel_i + kernel_j) * 0.5f;

            // We do double updates under particle exchange because we are called once per pair
            // and both the fact of the interaction and the kernel contribution are symmetric.
            kernelContributionWriter.TryAdd(new EntityOrderedPair(i, j), kernel);
            kernelContributionWriter.TryAdd(new EntityOrderedPair(j, i), new float4(- kernel.xyz, kernel.w));

            interactionPairsWriter.Add(i, j);
            interactionPairsWriter.Add(j, i);

            //Debug.Log("Trigger on frame number " + frameNumber.ToString() +
            //    " between A: " + triggerEvent.EntityA.ToString() +
            //    " and B: " + triggerEvent.EntityB.ToString());
        }
    }
#endif
}