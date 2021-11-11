#define LIST_OF_QUEUES
//#define COMMAND_BUFFER
//#define RECORD_ALL_COLLISIONS 
// Debug flag to help understand what is being detected as
// a collision in simple situations, verify correctness

using System;
using UnityEngine;
using Unity.Entities;
using Unity.Physics.Systems;
using Unity.Physics;
using Unity.Jobs;
using Unity.Transforms;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using static Unity.Physics.DispatchPairSequencer;
using UnityEngine.Assertions;

#if RECORD_ALL_COLLISIONS
[InternalBufferCapacity(8)]
public struct ParticleCollision : IBufferElementData
{
    public Entity Other;
    public float distance;
}
#endif

// Copied from IPhysicsSystem because this is a good idea generally
// for systems that have to run in-order repeatedly and handle their
// own data structure cleanup
interface IParticleSystem
{
    void AddInputDependency(JobHandle inputDep);

    JobHandle GetOutputDependency();
}

[BurstCompile]
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(BuildPhysicsWorld))]
[UpdateBefore(typeof(StepPhysicsWorld))]
public class KernelSystem : SystemBase, IParticleSystem
{
    private StepPhysicsWorld m_StepPhysicsWorld;
    private NativeStream m_interactions;

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
#if RECORD_ALL_COLLISIONS
            var collisionCount = 0;
#endif
            // Block the main thread until completion here
            Entities.ForEach((in DynamicBuffer<ParticleInteraction> interactions
#if RECORD_ALL_COLLISIONS
                ,in DynamicBuffer < ParticleCollision > collisions
#endif
                ) =>
            {
                interactionCount += interactions.Length;
#if RECORD_ALL_COLLISIONS
                collisionCount += collisions.Length;
#endif
                particleCount += 1;
            }).Run();

            Debug.Log("Update #" + updateNumber.ToString());
            Debug.Log("Interaction Pairs: " + interactionCount.ToString());
            Debug.Log("Particle count: " + particleCount.ToString());
            Debug.Log("Avg interactions per particle: " + (interactionCount / particleCount).ToString());
#if RECORD_ALL_COLLISIONS
            Debug.Log("Collision Pairs: " + collisionCount.ToString());
            Debug.Log("Avg collisions per particle: " + (collisionCount / particleCount).ToString());
#endif
        }

        // Go over every particle and remove all the kernel contributions.
        // On the main thread of course!
        // This is not as bad as it seems, because Clear leaves memory capacity allocated.
        Entities.ForEach((ref DynamicBuffer<ParticleInteraction> interactions
#if RECORD_ALL_COLLISIONS
            ,ref DynamicBuffer<ParticleCollision> collisions
#endif
            ) =>
        {
            interactions.Clear();
#if RECORD_ALL_COLLISIONS
            collisions.Clear();
#endif
        }).Run();

        if (m_interactions.IsCreated) {
            m_interactions.Dispose();
        }

        // Reset our input Dependency, so we can start collecting for the next frame
        inputDependency = default;
    }

    unsafe protected override void OnUpdate()
    {
        Cleanup();
#if true
        m_StepPhysicsWorld.EnqueueCallback(SimulationCallbacks.Phase.PostBroadphase, 
            (ref ISimulation sim, ref PhysicsWorld pw, JobHandle inputDeps) =>
            {
                // A little paranoid since we're a custom callback that only exists in modified unity physics
                if (sim.Type != SimulationType.UnityPhysics)
                {
                    throw new NotImplementedException("SPH KernelSystem Only works with Unity Physics");
                }
                var dynamicVsDynamicBodyPairs = ((Simulation)sim).StepContext.dynamicVsDynamicBodyPairs;

                // First filter this list such that we
                // 1) Hand back to physics any pairs that are not SPH particles
                // 2) Retain only SPH particles which have non-zero kernel interaction

                NativeArray<int> physicsPairsForEachCount0 = new NativeArray<int>(1, Allocator.TempJob);
                var filterPairsPrePass = new StreamForEachCountJob()
                {
                    Stream = dynamicVsDynamicBodyPairs,
                    ForEachCount0 = physicsPairsForEachCount0
                };
                inputDeps = filterPairsPrePass.Schedule(inputDeps);

                // This stream goes back out to physics
                // We do not dispose it, that will be Physics's job
                NativeStream dynamicVsDynamicBodyPairsOut;
                JobHandle dynamicVsDynamicBodyPairsOutHandle = NativeStream.ScheduleConstruct(
                    out dynamicVsDynamicBodyPairsOut,
                    physicsPairsForEachCount0, inputDeps, Allocator.TempJob);

                // Put it where physics will find it
                ((Simulation)sim).StepContext.dynamicVsDynamicBodyPairs = dynamicVsDynamicBodyPairsOut;

                // This one id for us, and is destroyed at the end
                NativeStream particleBodyPairsStream;
                JobHandle particleBodyPairsStreamHandle = NativeStream.ScheduleConstruct(out particleBodyPairsStream,
                    physicsPairsForEachCount0, inputDeps, Allocator.TempJob);

                inputDeps = JobHandle.CombineDependencies(dynamicVsDynamicBodyPairsOutHandle, particleBodyPairsStreamHandle);

                var filterPairs = new FilterPairs()
                {
                    // Input
                    Bodies = pw.Bodies,
                    TranslationData = GetComponentDataFromEntity<Translation>(true),
                    SmoothingData = GetComponentDataFromEntity<ParticleSmoothing>(true),
                    DynamicVsDynamicPairs = dynamicVsDynamicBodyPairs,

                    // Output
                    PhysicsBodyPairsWriter = dynamicVsDynamicBodyPairsOut.AsWriter(),
                    ParticleBodyPairsWriter = particleBodyPairsStream.AsWriter()
                };

                inputDeps = filterPairs.ScheduleUnsafeIndex0(physicsPairsForEachCount0, 1, inputDeps);

                // Flatten the particle pairs stream down to a list, so we can sort and process
                var unsortedPairs = new NativeList<DispatchPair>(Allocator.TempJob);
                var elementOffsets = new NativeList<int>(Allocator.TempJob);
                var prePass = new FlattenPairsPrePass()
                {
                    DynamicVsDynamicPairs = particleBodyPairsStream,
                    UnsortedPairs = unsortedPairs,
                    ElementOffsets = elementOffsets,
                };
                inputDeps = prePass.Schedule(inputDeps);

                var flattenPairs = new FlattenPairs()
                {
                    DynamicVsDynamicPairs = particleBodyPairsStream,
                    ElementOffsets = elementOffsets,
                    UnsortedPairs = unsortedPairs,
                };
                inputDeps = flattenPairs.Schedule(elementOffsets, 1, inputDeps);

                // Now make two sorted lists one by bodyA the other by body B
                int numBodies = pw.Bodies.Length;
                Assert.AreNotEqual(numBodies, 0);
                JobHandle sortedByBodyAPairsHandle = ScheduleSortPairsJob(true, unsortedPairs, numBodies,
                    out NativeList<DispatchPair> sortedByBodyAPairs,
                    out NativeList<int> bodyAOffsets, 
                    inputDeps);

                JobHandle sortedByBodyBPairsHandle = ScheduleSortPairsJob(false, unsortedPairs, numBodies,
                    out NativeList<DispatchPair> sortedByBodyBPairs,
                    out NativeList<int> bodyBOffsets, 
                    inputDeps);

                inputDeps = JobHandle.CombineDependencies(sortedByBodyAPairsHandle, sortedByBodyBPairsHandle);

                // Calculate all of the one-way kernel interactions, going through both arrays
                // Multithreading is per-particle, so we can immediately write the particle's DynamicBuffer component
                inputDeps = ScheduleCalculateInteractionJob(
                    ref sortedByBodyAPairs, ref sortedByBodyBPairs,
                    ref bodyAOffsets, ref bodyBOffsets, 
                    pw.Bodies,
                    GetComponentDataFromEntity<ParticleSmoothing>(true),
                    GetComponentDataFromEntity<Translation>(true),
                    GetBufferFromEntity<ParticleInteraction>(),
                    inputDeps);

                // Schedule cleanup of data structures
                // TODO parallelize this using Dispose handles pattern.

                // We dispose this because we don't need it anymore and also physics doesn't need it
                // Physics should only have the "out" pairs stream constructed during filtering
                inputDeps = dynamicVsDynamicBodyPairs.Dispose(inputDeps);

                inputDeps = physicsPairsForEachCount0.Dispose(inputDeps);
                inputDeps = particleBodyPairsStream.Dispose(inputDeps);
                inputDeps = unsortedPairs.Dispose(inputDeps);
                inputDeps = elementOffsets.Dispose(inputDeps);
                inputDeps = bodyAOffsets.Dispose(inputDeps);
                inputDeps = bodyBOffsets.Dispose(inputDeps);
                inputDeps = sortedByBodyAPairs.Dispose(inputDeps);
                inputDeps = sortedByBodyBPairs.Dispose(inputDeps);

                var scheduledJobHandle = JobHandle.CombineDependencies(inputDeps, Dependency);
                outputDependency = scheduledJobHandle;
                return scheduledJobHandle;
            }, Dependency);
#else
                m_StepPhysicsWorld.EnqueueCallback( SimulationCallbacks.Phase.PostCreateDispatchPairs,
            (ref ISimulation sim, ref PhysicsWorld pw, JobHandle inputDeps) =>
            {

                // Throw if we can't have unity Physics, we rely on unity physics internal data structures
                if (sim.Type != SimulationType.UnityPhysics)
                {
                    throw new NotImplementedException("SPH KernelSystem Only works with Unity Physics");
                }
                var stepContext = ((Simulation)sim).StepContext;
                var phasedDispatchPairs = stepContext.PhasedDispatchPairs;
                var solverSchedulerInfo = stepContext.SolverSchedulerInfo;

                NativeArray<int> numWorkItems = solverSchedulerInfo.NumWorkItems;

                inputDeps = NativeStream.ScheduleConstruct(out m_interactions, numWorkItems, inputDeps, Allocator.TempJob);

                // Job to evaluate kernel across all interaction pairs and push into native queues
                int batchCount = 1; // How many interaction pairs should a worker process before returning to the pool?
                var capturePairsTest = new InteractionPairsJob()
                {
                    phasedDispatchPairs = phasedDispatchPairs.AsDeferredJobArray(),
                    bodies = pw.Bodies,
                    smoothingData = GetComponentDataFromEntity<ParticleSmoothing>(true), // RO access to particle smoothing size
                    translationData = GetComponentDataFromEntity<Translation>(true),
                    interactionsWriter = m_interactions.AsWriter(),
                    SolverSchedulerInfo = solverSchedulerInfo,
                };
                inputDeps = capturePairsTest.ScheduleUnsafeIndex0(numWorkItems, batchCount, inputDeps);
                // Job to disable further physics calculation on our pairs
                var disablePairs = new DisablePairsJob();
                inputDeps = disablePairs.Schedule(sim, ref pw, inputDeps);
                // Iterate multithreaded over m_interactions and copy to particle
                // dynamic buffer components
                inputDeps = ScheduleCopyInteractionJob(ref m_interactions, ref solverSchedulerInfo,
                    GetBufferFromEntity<ParticleInteraction>(), 
                    inputDeps);
                var scheduledJobHandle = JobHandle.CombineDependencies(inputDeps, Dependency);
                outputDependency = scheduledJobHandle;
                return scheduledJobHandle;
           }, Dependency);
#endif
           updateNumber++;
    }

    private JobHandle ScheduleCalculateInteractionJob(
        ref NativeList<DispatchPair> sortedByBodyAPairs, 
        ref NativeList<DispatchPair> sortedByBodyBPairs, 
        ref NativeList<int> bodyAOffsets, 
        ref NativeList<int> bodyBOffsets,
        NativeArray<RigidBody> bodies,
        ComponentDataFromEntity<ParticleSmoothing> smoothingData,
        ComponentDataFromEntity<Translation> translationData,
        BufferFromEntity<ParticleInteraction> bufferLookup,

        JobHandle handle)
    {
        var calculateInteractionJob = new CalculateInteractionJob
        {
            SortedByBodyAPairs = sortedByBodyAPairs,
            SortedByBodyBPairs = sortedByBodyBPairs,
            BodyAOffsets = bodyAOffsets,
            BodyBOffsets = bodyBOffsets,
            Bodies = bodies,
            SmoothingData = smoothingData,
            TranslationData = translationData,
            BufferLookup = bufferLookup,
        };
        handle = calculateInteractionJob.Schedule(bodyAOffsets, 1, handle);
        return handle;
    }
    [BurstCompile]
    public struct CalculateInteractionJob : IJobParallelForDefer
    {
        // Inputs
        [ReadOnly] public NativeList<DispatchPair> SortedByBodyAPairs;
        [ReadOnly] public NativeList<DispatchPair> SortedByBodyBPairs;
        [ReadOnly] public NativeList<int> BodyAOffsets;
        [ReadOnly] public NativeList<int> BodyBOffsets;
        [ReadOnly] public NativeArray<RigidBody> Bodies;
        [ReadOnly] public ComponentDataFromEntity<ParticleSmoothing> SmoothingData;
        [ReadOnly] public ComponentDataFromEntity<Translation> TranslationData;

        // Output
        [NativeDisableParallelForRestriction]
        public BufferFromEntity<ParticleInteraction> BufferLookup;
        public void Execute(int bodyIndex)
        {
            Assert.AreEqual(BodyAOffsets.Length, Bodies.Length);
            Assert.AreEqual(BodyBOffsets.Length, Bodies.Length);
            Assert.AreNotEqual(Bodies.Length, 0);
            Assert.AreEqual(SortedByBodyAPairs.Length, SortedByBodyBPairs.Length);
            Assert.AreNotEqual(SortedByBodyAPairs.Length, 0);
            Entity i = Bodies[bodyIndex].Entity;
            if( ! BufferLookup.HasComponent(i) || 
                ! SmoothingData.HasComponent(i) || 
                ! TranslationData.HasComponent(i) ) {
                // Debug.Log("Found weird Entity " + i.Index + " RigidBody Index: " + bodyIndex);
                return;
            }
            DynamicBuffer<ParticleInteraction> buffer = BufferLookup[i];

            //insert interactions where this body was first
            int BodyAPastEndOffset = ((bodyIndex + 1) < BodyAOffsets.Length) ? 
                BodyAOffsets[bodyIndex + 1] : SortedByBodyAPairs.Length;
            int BodyAStartOffset = BodyAOffsets[bodyIndex];
            for( int index = BodyAStartOffset; index < BodyAPastEndOffset; index++)
            {
                Entity j = Bodies[SortedByBodyAPairs[index].BodyIndexB].Entity;
                ParticleInteraction interaction = CalculateInteraction(i, j);
                if (interaction.KernelSymmetric.w > 0.0f)
                {
                    buffer.Add(interaction);
                }
            }

            //insert interactions where this body was second
            int BodyBPastEndOffset = ((bodyIndex + 1) < BodyBOffsets.Length) ?
                BodyBOffsets[bodyIndex + 1] : SortedByBodyBPairs.Length;
            int BodyBStartOffset = BodyBOffsets[bodyIndex];
            for (int index = BodyBStartOffset; index < BodyBPastEndOffset; index++)
            {
                Entity j = Bodies[SortedByBodyBPairs[index].BodyIndexA].Entity;
                ParticleInteraction interaction = CalculateInteraction(i, j);
                if (interaction.KernelSymmetric.w > 0.0f)
                {
                    buffer.Add(interaction);
                }
            }
        }
        public ParticleInteraction CalculateInteraction(Entity i, Entity j)
        {
            var r_i = TranslationData[i].Value;
            var r_j = TranslationData[j].Value;

            var distance = math.length(r_i - r_j);

            // Calculate symmetrized kernel contribution.
            // Once for derivatives w.r.t. particle i position, again for derivitives w.r.t. particle j
            // TODO: There is some duplicaiton of effort here especially on the kernel values
            // TODO: It is possible to reject pairs before performing this calculation

            // Derivatives are implicitly w.r.t to particle i position.
            // therefore the "symmetrized" kernel still has a preferred index in the derivative
            float4 kernel_ij_i = SplineKernel.KernelAndGradienti(r_i, r_j, SmoothingData[i].h);
            float4 kernel_ij_j = SplineKernel.KernelAndGradienti(r_i, r_j, SmoothingData[j].h);
            var kernel_ij_sym = (kernel_ij_i + kernel_ij_j) * 0.5f;
            var ij = new ParticleInteraction
            {
                Self = i,
                Other = j,
                KernelThis = kernel_ij_i,
                KernelOther = kernel_ij_j,
                KernelSymmetric = kernel_ij_sym,
                distance = distance,
            };
            return ij;
        }
    }

    private static JobHandle ScheduleSortPairsJob(bool bodyAIsKey, NativeList<DispatchPair> unsortedPairs, 
        int numBodies, 
        out NativeList<DispatchPair> sortedPairs, 
        out NativeList<int> bodyOffsets,
        JobHandle handle, bool singleThread = true)
    {
        // TODO: Optimizations
        // - See if single thread radix sort (like in physics) is faster here.
        //   Downside... whole rest of paralllel algorithm is waiting on
        //   this ordering
        // - Choose a group of indicies (masking off some low bits) and
        //   then assign that group of rigid bodies to the same radix bucket
        sortedPairs = new NativeList<DispatchPair>(Allocator.TempJob);
        bodyOffsets = new NativeList<int>(numBodies, Allocator.TempJob);
        var bodyLengths = new NativeArray<int>(numBodies, Allocator.TempJob);

        if (singleThread)
        {
            Assert.AreNotEqual(numBodies, 0);
            var currentBodyOffsets = new NativeArray<int>(numBodies, Allocator.TempJob);

            var sortJobST = new SortPairsSTJob
            {
                BodyAIsKey = bodyAIsKey,
                UnsortedPairs = unsortedPairs,
                BodyLengths = bodyLengths,
                CurrentBodyOffsets = currentBodyOffsets,
                BodyOffsets = bodyOffsets,
                SortedPairs = sortedPairs,
                NumBodies = numBodies,
            };
            handle = sortJobST.Schedule(handle);

            handle = currentBodyOffsets.Dispose(handle);
        }
        else
        {
            var generateOffsets = new GenerateOffsetsJob()
            {
                BodyAIsKey = bodyAIsKey,
                UnsortedPairs = unsortedPairs,
                SortedPairs = sortedPairs,
                BodyOffsets = bodyOffsets,
                BodyLengths = bodyLengths,
                NumBodies = numBodies,
            }; 
            handle = generateOffsets.Schedule(handle);

            var sortJob = new SortPairsJob
            {
                BodyAIsKey = bodyAIsKey,
                UnsortedPairs = unsortedPairs,
                BodyOffsets = bodyOffsets,
                SortedPairs = sortedPairs,
            };

            handle = sortJob.Schedule(bodyOffsets, 1, handle);
        }

        // Side effect, our consumers wait on us finishing to dispose of our data
        // TODO look at SimulationJobHandles, there seems to be a solution pattern for this.
        handle = bodyLengths.Dispose(handle);

        return handle;
    }


    [BurstCompile]
    public struct SortPairsSTJob : IJob
    {
        // In
        [ReadOnly] public bool BodyAIsKey;  // True if we're using body A as key
        [ReadOnly] public int NumBodies; // Needed for length calculations
        [ReadOnly] public NativeList<DispatchPair> UnsortedPairs; // We're iterating over this

        //Scratch space
        public NativeArray<int> CurrentBodyOffsets;

        // Out: Offset and length in the sorted pairs array organized by body index
        public NativeList<int> BodyOffsets;
        public NativeArray<int> BodyLengths;

        // Out: Sorted array by whichever key was selected by BodyAIsKey
        public NativeList<DispatchPair> SortedPairs;

        public void Execute()
        {
            SortedPairs.ResizeUninitialized(UnsortedPairs.Length);
            BodyOffsets.ResizeUninitialized(NumBodies);

            Assert.AreEqual(UnsortedPairs.Length, SortedPairs.Length);
            Assert.AreEqual(NumBodies, BodyOffsets.Length);
            
            // I think this is unnecessary to check because of a quirk in NativeArray
            //Assert.AreEqual(NumBodies, BodyLengths.Length);
            // Count the bodies by index

            for (int i = 0; i < UnsortedPairs.Length; i++)
            {
                var pair = UnsortedPairs[i];
                var bodyIndex = BodyAIsKey ? pair.BodyIndexA : pair.BodyIndexB;
                BodyLengths[bodyIndex] += 1;
            }

            // Calculate offsets in a sorted array
            int totalBodies = 0;
            for (int i = 0; i < NumBodies; i++)
            {
                BodyOffsets[i] = totalBodies;
                CurrentBodyOffsets[i] = totalBodies;
                totalBodies += BodyLengths[i];
            }

            // Copy elements into buckets based on the appropriate index

            for (int i = 0; i < UnsortedPairs.Length; i++)
            {
                DispatchPair pair = UnsortedPairs[i];
                int bodyIndex = BodyAIsKey ? pair.BodyIndexA : pair.BodyIndexB;
                int sortedIndex = CurrentBodyOffsets[bodyIndex];
                CurrentBodyOffsets[bodyIndex]++;
                SortedPairs[sortedIndex] = pair;
            }
        }
    }

    [BurstCompile]
    public struct SortPairsJob: IJobParallelForDefer
    {
        // Input
        [ReadOnly] public bool BodyAIsKey;
        [ReadOnly] public NativeList<int> BodyOffsets;
        [ReadOnly] public NativeList<DispatchPair> UnsortedPairs;

        // Output
        [NativeDisableParallelForRestriction]
        public NativeList <DispatchPair> SortedPairs;
        public void Execute(int bodyIndex)
        {
            // Lookup offset in body index
            int offset = BodyOffsets[bodyIndex];
            
            // Go across entire unsorted body pairs array looking for our particle
            // Copy to appropriate location in sorted pairs array
            for (int i = 0; i< UnsortedPairs.Length; i++)
            {
                var key = BodyAIsKey ? UnsortedPairs[i].BodyIndexA : UnsortedPairs[i].BodyIndexB;
                if (key == bodyIndex)
                {
                    SortedPairs[offset] = UnsortedPairs[i];
                    offset++;
                }
            }
        }
    }

    [NoAlias]
    [BurstCompile]
    public struct GenerateOffsetsJob: IJob
    {
        // In
        [ReadOnly] public bool BodyAIsKey;  // True if we're using body A as key
        [ReadOnly] public int NumBodies; // Needed for length calculations
        [ReadOnly] public NativeList<DispatchPair> UnsortedPairs; // We're iterating over this

        // Allocate this
        public NativeList<DispatchPair> SortedPairs;

        // Out: Offset and length in the sorted pairs array organized by body index
        public NativeList<int> BodyOffsets;
        public NativeArray<int> BodyLengths;

        public void Execute() 
        {
            SortedPairs.ResizeUninitialized(UnsortedPairs.Length);
            BodyOffsets.ResizeUninitialized(NumBodies);
            // Count the bodies by index
            for (int i = 0; i < UnsortedPairs.Length; i++)
            {
                var pair = UnsortedPairs[i];
                var bodyIndex = BodyAIsKey ? pair.BodyIndexA : pair.BodyIndexB;
                BodyLengths[bodyIndex] += 1;                
            }

            // Calculate offsets in a sorted array
            int totalBodies = 0;
            for (int i = 0; i < NumBodies; i++)
            {
                BodyOffsets[i] = totalBodies;
                totalBodies += BodyLengths[i];
            }
        }
    }

    // Run over the For-Each index of the nativeStream counting elements
    // Put the cumulative element counts into an array
    [BurstCompile]
    [NoAlias]
    public struct FlattenPairsPrePass : IJob
    {
        // Input
        [ReadOnly] public NativeStream DynamicVsDynamicPairs;

        // Allocate correct size
        [NativeDisableParallelForRestriction]
        public NativeList<DispatchPair> UnsortedPairs;

        // Output
        // Array offsets indexed by the foreach index of DynamicVsDynamicPairs
        public NativeList<int> ElementOffsets;
        

        public void Execute()
        {
            int forEachCount = DynamicVsDynamicPairs.ForEachCount;

            ElementOffsets.ResizeUninitialized(forEachCount);
            var reader = DynamicVsDynamicPairs.AsReader();
            int totalElements = 0;
            for (int i = 0; i < forEachCount; i++)
            {
                reader.BeginForEachIndex(i);
                ElementOffsets[i] = totalElements;
                totalElements += reader.RemainingItemCount;
            }
            UnsortedPairs.ResizeUninitialized(totalElements);
        }

        public static unsafe bool StreamEq(ref NativeStream a, ref NativeStream b)
        {
            return UnsafeUtility.AddressOf(ref a) == UnsafeUtility.AddressOf(ref b);
        }
    }


    // Counts the elements in NativeStream at job-time.
    // Output is a single-element NativeArray that works with IJobParallelForDefer Scheduling functions
    [BurstCompile]
    public struct StreamForEachCountJob : IJob
    {
        // Input
        [ReadOnly] public NativeStream Stream;
        
        // Output
        public NativeArray<int> ForEachCount0;

        public void Execute()
        {
            ForEachCount0[0] = Stream.ForEachCount;
        }
    }


    // Assemble an array from the NativeStream
    // Use the element counts to assemble the array in paralllel
    // Each thread handles a single foreach index of the native stream
    [BurstCompile]
    public struct FilterPairs : IJobParallelForDefer
    {
        // Input
        [ReadOnly] public NativeArray<RigidBody> Bodies;
        [ReadOnly] public ComponentDataFromEntity<Translation> TranslationData;
        [ReadOnly] public ComponentDataFromEntity<ParticleSmoothing> SmoothingData;
        [ReadOnly] public NativeStream DynamicVsDynamicPairs;  // Stream is composed of BodyIndexPair only

        // Output:
        // These are streams of BodyIndexPair
        public NativeStream.Writer PhysicsBodyPairsWriter; // Destined for Physics
        public NativeStream.Writer ParticleBodyPairsWriter;  // Destined for further SPH processing

        public void Execute(int forEachIndex)
        {
            var reader = DynamicVsDynamicPairs.AsReader();

            reader.BeginForEachIndex(forEachIndex);
            PhysicsBodyPairsWriter.BeginForEachIndex(forEachIndex);
            ParticleBodyPairsWriter.BeginForEachIndex(forEachIndex);
            while (reader.RemainingItemCount > 0)
            {
                BodyIndexPair bodyIndexPair = reader.Read<BodyIndexPair>();
                Entity entityA = Bodies[bodyIndexPair.BodyIndexA].Entity;
                Entity entityB = Bodies[bodyIndexPair.BodyIndexB].Entity;

                if( SmoothingData.HasComponent(entityA) && SmoothingData.HasComponent(entityB))
                {
                    var r_i = TranslationData[entityA].Value;
                    var r_j = TranslationData[entityB].Value;
                    var h_i = SmoothingData[entityA].h;
                    var h_j = SmoothingData[entityB].h;

                    if (SplineKernel.Interacts(r_i, r_j, h_i, h_j))
                    {
                        // This particle has nonzero neighbor interaction
                        ParticleBodyPairsWriter.Write(bodyIndexPair);
                    }
                } 
                else 
                {
                    // Not ours, output to the list physics will consume
                    PhysicsBodyPairsWriter.Write(bodyIndexPair);
                }
            }
            ParticleBodyPairsWriter.EndForEachIndex();
            PhysicsBodyPairsWriter.EndForEachIndex();
            reader.EndForEachIndex();
        }
    }

    // Assemble an array from the NativeStream
    // Use the element counts to assemble the array in paralllel
    // Each thread handles a single foreach index of the native stream
    [BurstCompile]
    public struct FlattenPairs : IJobParallelForDefer
    {
        // Input
        [ReadOnly] public NativeStream DynamicVsDynamicPairs;
        [ReadOnly] public NativeList<int> ElementOffsets;

        // Output
        [NativeDisableParallelForRestriction]
        public NativeList<DispatchPair> UnsortedPairs;

        public void Execute(int forEachIndex)
        {
            var offset = ElementOffsets[forEachIndex];
            var reader = DynamicVsDynamicPairs.AsReader();
         
            reader.BeginForEachIndex(forEachIndex);
            while(reader.RemainingItemCount > 0)
            {
                DispatchPair pair = DispatchPair.CreateCollisionPair(reader.Read<BodyIndexPair>());
                UnsortedPairs[offset] = pair;
                offset++;
            }
            reader.EndForEachIndex();
        }
    }

    // Copying the pattern of the Jacobian solver which also needs to write
    // A per-body array of Motions. Our version is appending to a dynamic component
    // But the requirement that we be able to only have a single concurrent thread
    // accessing a body is the same.
    //
    // How this works:
    // Within a Phase, a body appearing in one work item means all of its interactions
    // wil be in the same work batch. 
    // 
    // By default, the physics scheduler uses a batch size of 8 interactions and 16 phases. 
    // The last phase is special and has to be run single-threaded. When the number of 
    // interactions per particle exceeds 8*15=120 we begin to spill over into this special phase
    // and losing speedup from multithreading.
    //
    // Physics is doing a masking calculation to choose which phase a body goes in. That masking
    // uses the bits of a ushort. This makes sense in the context of most games where
    // high-collisions-per-dynamic-body is atypical. 
    //
    // If you are running into the limit with collisions, the simplest way forward is to increase
    // the batch size. Unfortunately, with a dense graph of collisions each phase gets dominated
    // by a small number of particles. This means increasing the batch size still results in most
    // collisions being sorted to the single threaded final phase.
    //
    // TL;DR: This does not achieve a multithreaded speedup. What is needed is a way to iterate
    // over the collisions per-particle, and have each thread only work on a single particle.
    
    internal static unsafe JobHandle ScheduleCopyInteractionJob(
        ref NativeStream interactions,
        ref DispatchPairSequencer.SolverSchedulerInfo solverSchedulerInfo,
        BufferFromEntity<ParticleInteraction> bufferLookup,
        JobHandle inputDeps)
    {
        JobHandle handle = inputDeps;
        int numPhases = solverSchedulerInfo.NumPhases;

        var phaseInfoPtrs = (DispatchPairSequencer.SolverSchedulerInfo.SolvePhaseInfo*) 
            NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(solverSchedulerInfo.PhaseInfo);

        for (int phaseId = 0; phaseId < numPhases; phaseId++)
        {
            var copyInteractionJob = new CopyInteractionJob()
            {
                interactionsReader = interactions.AsReader(),
                bufferLookup = bufferLookup,
                PhaseIndex = phaseId,
                Phases = solverSchedulerInfo.PhaseInfo,
            };

            // NOTE: The last phase must be executed on a single job since 
            //       The last phase is were the sorting algorithm dumps all dispatch
            //       pairs that have been fully excluded from other phases
            // Copying use of int.MaxValue/2 from similar physics code (ScheduleSolveJacobiansJobs) in Solver.cs
            bool isLastPhase = phaseId == numPhases - 1;
            int batchSize = isLastPhase ? (int.MaxValue / 2) : 1;

            int* numWorkItems = &(phaseInfoPtrs[phaseId].NumWorkItems);
            handle = copyInteractionJob.Schedule(numWorkItems, batchSize, handle);
        }

        return handle;
    }
    
    // This job nerfs future physics on any interaction pair identified in 
    // Broadphase which we will be computing. Operates in concert with InteractionPairsJob
    [BurstCompile]
    public struct DisablePairsJob : IBodyPairsJob
    {
        public void Execute(ref ModifiableBodyPair triggerEvent)
        {
            // TODO: Filter object interactions so we only operate on particles

            triggerEvent.Disable();
            // Stop physics from doing more with this collision!
            // no jacobians! No collision response!
            // only step-velocity integration!
        }
    }

    [NoAlias]
    [BurstCompile]
    public struct CopyInteractionJob : IJobParallelForDefer
    {
        [ReadOnly, NoAlias] public NativeStream.Reader interactionsReader;

        [NativeDisableParallelForRestriction]
        public BufferFromEntity<ParticleInteraction> bufferLookup;

        [ReadOnly, NoAlias] public NativeArray<DispatchPairSequencer.SolverSchedulerInfo.SolvePhaseInfo> Phases;
        public int PhaseIndex;

        public void Execute(int workItemIndex)
        {
            int workItemStartIndexOffset = Phases[PhaseIndex].FirstWorkItemIndex;
            ExecuteImpl(workItemIndex + workItemStartIndexOffset);
        }

        public void ExecuteImpl(int workItemIndex)
        {
            interactionsReader.BeginForEachIndex(workItemIndex);
            while(interactionsReader.RemainingItemCount > 0)
            {
                ParticleInteraction interaction = interactionsReader.Read<ParticleInteraction>();
                DynamicBuffer<ParticleInteraction> buffer = bufferLookup[interaction.Self];
                /*
                Debug.Log(workItemIndex + " " +
                    interaction.Self.Index + " " +
                    interaction.Other.Index + " " + 
                    interaction.distance);
                */
                buffer.Add(interaction);
            }
            interactionsReader.EndForEachIndex();
        }
    }

    
    
    
    [BurstCompile]
    [NoAlias]
    public struct InteractionPairsJob : IJobParallelForDefer
    {
        [ReadOnly, NoAlias] public NativeArray<DispatchPairSequencer.DispatchPair> phasedDispatchPairs;

        [ReadOnly, NoAlias] public NativeArray<RigidBody> bodies;
        [ReadOnly] public ComponentDataFromEntity<ParticleSmoothing> smoothingData;
        [ReadOnly] public ComponentDataFromEntity<Translation> translationData;
        [ReadOnly, NoAlias] public DispatchPairSequencer.SolverSchedulerInfo SolverSchedulerInfo;
        [WriteOnly, NoAlias] public NativeStream.Writer interactionsWriter;

        public void Execute(int workItemIndex)
        {
            int dispatchPairReadOffset = SolverSchedulerInfo.GetWorkItemReadOffset(workItemIndex, out int numPairsToRead);
            interactionsWriter.BeginForEachIndex(workItemIndex);

            ExecuteImpl(dispatchPairReadOffset, numPairsToRead);

            interactionsWriter.EndForEachIndex();
        }
        public void ExecuteImpl(int dispatchPairReadOffset, int numPairsToRead)
        {
            for (int index = 0; index < numPairsToRead; index++)
            {
                DispatchPairSequencer.DispatchPair dispatchPair = phasedDispatchPairs[dispatchPairReadOffset + index];
                // Skip joint pairs and invalid pairs
                if (dispatchPair.IsJoint || !dispatchPair.IsValid)
                {
                    return;
                }

                Entity i = bodies[dispatchPair.BodyIndexA].Entity;
                Entity j = bodies[dispatchPair.BodyIndexB].Entity;

                var r_i = translationData[i].Value;
                var r_j = translationData[j].Value;

                var distance = math.length(r_i - r_j);

                // Calculate symmetrized kernel contribution.
                // Once for derivatives w.r.t. particle i position, again for derivitives w.r.t. particle j
                // TODO: There is some duplicaiton of effort here especially on the kernel values
                // TODO: It is possible to reject pairs before performing this calculation

                // Derivatives are implicitly w.r.t to particle i position.
                // therefore the "symmetrized" kernel still has a preferred index in the derivative
                float4 kernel_ij_i = SplineKernel.KernelAndGradienti(r_i, r_j, smoothingData[i].h);
                float4 kernel_ij_j = SplineKernel.KernelAndGradienti(r_i, r_j, smoothingData[j].h);
                var kernel_ij_sym = (kernel_ij_i + kernel_ij_j) * 0.5f;
                var ij = new ParticleInteraction
                {
                    Self = i,
                    Other = j,
                    KernelThis = kernel_ij_i,
                    KernelOther = kernel_ij_j,
                    KernelSymmetric = kernel_ij_sym,
                    distance = distance,
                };

                // The only difference here is the derivatives which are now w.r.t. particle j's position
                // for an isotropic and even kernel function it should be true that
                // kernel_ji_sym = new float4( -kernel_ij_sym.xyz, kernel_ij_sym.w)
                float4 kernel_ji_i = SplineKernel.KernelAndGradienti(r_j, r_i, smoothingData[i].h);
                float4 kernel_ji_j = SplineKernel.KernelAndGradienti(r_j, r_i, smoothingData[j].h);
                var kernel_ji_sym = (kernel_ji_i + kernel_ji_j) * 0.5f;
                var ji = new ParticleInteraction
                {
                    Self = j,
                    Other = i,
                    KernelThis = kernel_ji_j,
                    KernelOther = kernel_ji_i,
                    KernelSymmetric = kernel_ji_sym,
                    distance = distance,
                };

                // Only insert to data structures later steps use for iteration if there's
                // a weight greater than zero for this pair. This saves every later calculation
                // time in iteration.
                if (kernel_ij_sym.w > 0.0f)
                {
                    // We're going to insert into the buffers of i and j a record represeting
                    // The symmetrized kernel contribution of the other one

                    // Particle i
                    interactionsWriter.Write(ij);

                    // Particle j
                    interactionsWriter.Write(ji);
                }
#if RECORD_ALL_COLLISIONS
            // TODO fixup Will need parallel queue system if particlequeuewriters works

            // Reuse index from global order of collisions for when we record into ecb
            // This means we can only deal with 2^30 interactions max rather than 2^31 :(
            int index_i = index << 1;
            int index_j = index_i + 1;

            interactionDataWriter.AppendToBuffer(index_i, i, new ParticleCollision { Other = j , distance = math.length(r_i - r_j) });
            interactionDataWriter.AppendToBuffer(index_j, j, new ParticleCollision { Other = i , distance = math.length(r_i - r_j) });
#endif
            }
        }
    }
}