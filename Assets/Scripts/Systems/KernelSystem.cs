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
using static Unity.Physics.DispatchPairSequencer;
using Unity.Jobs.LowLevel.Unsafe;

[BurstCompile(OptimizeFor = OptimizeFor.Performance)]
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(BuildPhysicsWorld))]
[UpdateBefore(typeof(StepPhysicsWorld))]
public class KernelSystem : SystemBase, IPhysicsSystem
{
    private StepPhysicsWorld m_StepPhysicsWorld;
    private NativeStream m_interactions;

    public JobHandle GetOutputDependency() => outputDependency;
    private JobHandle outputDependency; // Only valid after StepPhysics runs

    // If you need to use the interaction data, register yourself so we 
    // wait for you to finish
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
            // Block the main thread until completion here
            Entities.ForEach((in DynamicBuffer<ParticleInteraction> interactions) =>
            {
                interactionCount += interactions.Length;
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

        if (m_interactions.IsCreated)
        {
            m_interactions.Dispose();
        }

        // Reset our input Dependency, so we can start collecting for the next frame
        inputDependency = default;
    }

    unsafe protected override void OnUpdate()
    {
        Cleanup();

        m_StepPhysicsWorld.EnqueueCallback(SimulationCallbacks.Phase.PostBroadphase,
            (ref ISimulation sim, ref PhysicsWorld pw, JobHandle inputDeps) =>
            {
                // A little paranoid since we're attaching to a custom callback that only exists
                // in modified unity physics
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
                // We do not dispose, because Physics will.
                NativeStream dynamicVsDynamicBodyPairsOut;
                JobHandle dynamicVsDynamicBodyPairsOutHandle = NativeStream.ScheduleConstruct(
                    out dynamicVsDynamicBodyPairsOut,
                    physicsPairsForEachCount0, inputDeps, Allocator.TempJob);

                ((Simulation)sim).StepContext.dynamicVsDynamicBodyPairs = dynamicVsDynamicBodyPairsOut;

                // This stream is pairs we will process.
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

                // We dispose this because we don't need it anymore and also physics doesn't need it
                // Physics should only have the dynamicVsDynamicBodyPairsOut stream constructed during 
                // filtering
                var disposeHandle1 = dynamicVsDynamicBodyPairs.Dispose(inputDeps);
                var disposeHandle2 = physicsPairsForEachCount0.Dispose(inputDeps);

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

                var disposeHandle3 = particleBodyPairsStream.Dispose(inputDeps);
                var disposeHandle4 = elementOffsets.Dispose(inputDeps);
                
                // Now make two sorted lists one by bodyA the other by body B
                int numBodies = pw.Bodies.Length;
                SimulationJobHandles sortedByAHandles = ScheduleSortPairsJob(true, unsortedPairs, numBodies,
                    out NativeList<DispatchPair> sortedByBodyAPairs,
                    out NativeList<int> bodyAOffsets,
                    inputDeps);

                SimulationJobHandles sortedByBHandles = ScheduleSortPairsJob(false, unsortedPairs, numBodies,
                    out NativeList<DispatchPair> sortedByBodyBPairs,
                    out NativeList<int> bodyBOffsets,
                    inputDeps);

                inputDeps = JobHandle.CombineDependencies(sortedByAHandles.FinalExecutionHandle, 
                    sortedByBHandles.FinalExecutionHandle);
                var disposeHandle5 = JobHandle.CombineDependencies(sortedByAHandles.FinalDisposeHandle, 
                    sortedByBHandles.FinalDisposeHandle);
                
                // Calculate all of the one-way kernel interactions, going through both arrays
                // Multithreading is per-particle, so this job immediately writes the
                // particle's DynamicBuffer component
                var calculateInteractionJob = new CalculateInteractionJob
                {
                    SortedByBodyAPairs = sortedByBodyAPairs,
                    SortedByBodyBPairs = sortedByBodyBPairs,
                    BodyAOffsets = bodyAOffsets,
                    BodyBOffsets = bodyBOffsets,
                    Bodies = pw.Bodies,
                    SmoothingData = GetComponentDataFromEntity<ParticleSmoothing>(true),
                    TranslationData = GetComponentDataFromEntity<Translation>(true),
                    BufferLookup = GetBufferFromEntity<ParticleInteraction>(),
                };
                inputDeps = calculateInteractionJob.Schedule(bodyAOffsets, 1, inputDeps);

                var disposeHandle6 = unsortedPairs.Dispose(inputDeps);
                var disposeHandle7 = bodyAOffsets.Dispose(inputDeps);
                var disposeHandle8 = bodyBOffsets.Dispose(inputDeps);
                var disposeHandle9 = sortedByBodyAPairs.Dispose(inputDeps);
                var disposeHandle10 = sortedByBodyBPairs.Dispose(inputDeps);

                JobHandle* deps = stackalloc JobHandle[10]
                {
                    disposeHandle1, disposeHandle2, disposeHandle3, disposeHandle4, disposeHandle5,
                    disposeHandle6, disposeHandle7, disposeHandle8, disposeHandle9, disposeHandle10,
                };
                var finalDisposeHandle = JobHandleUnsafeUtility.CombineDependencies(deps, 10);

                inputDeps = JobHandle.CombineDependencies(inputDeps, finalDisposeHandle);

                var scheduledJobHandle = JobHandle.CombineDependencies(inputDeps, Dependency);
                outputDependency = scheduledJobHandle;
                return scheduledJobHandle;
            }, Dependency);
        updateNumber++;
    }

    #region Kernel Calculation Job
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, DisableSafetyChecks = true)]
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
            Entity i = Bodies[bodyIndex].Entity;
            if (!BufferLookup.HasComponent(i) ||
                !SmoothingData.HasComponent(i) ||
                !TranslationData.HasComponent(i))
            {
                // Debug.Log("Found weird Entity " + i.Index + " RigidBody Index: " + bodyIndex);
                return;
            }
            DynamicBuffer<ParticleInteraction> buffer = BufferLookup[i];

            //insert interactions where this body was first
            int BodyAPastEndOffset = ((bodyIndex + 1) < BodyAOffsets.Length) ?
                BodyAOffsets[bodyIndex + 1] : SortedByBodyAPairs.Length;
            int BodyAStartOffset = BodyAOffsets[bodyIndex];
            for (int index = BodyAStartOffset; index < BodyAPastEndOffset; index++)
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

        // TODO: Memoize this every frame. We are guaranted to call CalculateInteraction 2x,
        //       once for i->j and once for j->i
        //
        //       The contributions differ by a minus sign in the gradient, so a lookup of either
        //       may be faster than recalculating.
        //
        //       One way: Use three arrays
        //       1) Array of ParticleInteraction indexed by unsorted pair index
        //       2) Array that goes from sorted-by-a index -> unsorted index 
        //       3) Array that goes from sorted-by-b index -> unsorted index
        //       2 & 3 are easy/fast to make from information available during sorting job.
        //
        //       Might be better cache-wise with two arrays:
        //       1) Array of ParticleInteraction indexed by sorted-by-a index  (a-sorted pass lookup directly here)
        //       2) Array that goes from sorted-by-a index -> sorted-by-b index (How do you make this array easily?)
        public ParticleInteraction CalculateInteraction(Entity i, Entity j)
        {
            var r_i = TranslationData[i].Value;
            var r_j = TranslationData[j].Value;
#if DEBUG_KERNEL
            var distance = math.length(r_i - r_j);
#endif
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
                Other = j,
                KernelThis = kernel_ij_i,
                KernelSymmetric = kernel_ij_sym,
#if DEBUG_KERNEL
                KernelOther = kernel_ij_j,
                Self = i,
                distance = distance,
#endif
            };
            return ij;
        }
    }
#endregion

#region Sorting
    private static SimulationJobHandles ScheduleSortPairsJob(bool bodyAIsKey, NativeList<DispatchPair> unsortedPairs,
        int numBodies,
        out NativeList<DispatchPair> sortedPairs,
        out NativeList<int> bodyOffsets,
        JobHandle handle, bool singleThread = true)
    {
        SimulationJobHandles handles = new SimulationJobHandles(handle);
        // TODO: Optimizations
        // - See if single thread radix sort (like in physics) is faster here.
        //   Downside... whole rest of paralllel algorithm is waiting on
        //   this ordering
        // - Choose a group of indicies (masking off some low bits) and
        //   then assign that group of rigid bodies to the same radix bucket
        sortedPairs = new NativeList<DispatchPair>(Allocator.TempJob);
        bodyOffsets = new NativeList<int>(numBodies, Allocator.TempJob);
        var bodyLengths = new NativeArray<int>(numBodies, Allocator.TempJob);

        JobHandle disposeHandle1 = default;

        if (singleThread)
        {
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

            disposeHandle1 = currentBodyOffsets.Dispose(handle);
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
        JobHandle disposeHandle2 = bodyLengths.Dispose(handle);

        handles.FinalExecutionHandle = handle;
        handles.FinalDisposeHandle = JobHandle.CombineDependencies(disposeHandle1, disposeHandle2);

        return handles;
    }


    [BurstCompile(OptimizeFor = OptimizeFor.Performance, DisableSafetyChecks = false)]
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

            // Count the bodies by index
            for (int i = 0; i < UnsortedPairs.Length; i++)
            {
                var pair = UnsortedPairs[i];
                var bodyIndex = BodyAIsKey ? pair.BodyIndexA : pair.BodyIndexB;
                BodyLengths[bodyIndex] += 1;
            }

            // Calculate start offsets for buckets
            int prev = BodyLengths[0];
            BodyOffsets[0] = 0;
            CurrentBodyOffsets[0] = 0;
            for (int i = 1; i < NumBodies; i++)
            {
                int current = BodyLengths[i];
                BodyOffsets[i] = BodyOffsets[i - 1] + prev;
                CurrentBodyOffsets[i] = CurrentBodyOffsets[i - 1] + prev;
                prev = current;
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
#region Multithreaded Sorting Jobs
    [BurstCompile]
    public struct SortPairsJob : IJobParallelForDefer
    {
        // Input
        [ReadOnly] public bool BodyAIsKey;
        [ReadOnly] public NativeList<int> BodyOffsets;
        [ReadOnly] public NativeList<DispatchPair> UnsortedPairs;

        // Output
        [NativeDisableParallelForRestriction]
        public NativeList<DispatchPair> SortedPairs;
        public void Execute(int bodyIndex)
        {
            // Lookup offset in body index
            int offset = BodyOffsets[bodyIndex];

            // Go across entire unsorted body pairs array looking for our particle
            // Copy to appropriate location in sorted pairs array
            for (int i = 0; i < UnsortedPairs.Length; i++)
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
    public struct GenerateOffsetsJob : IJob
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
#endregion
#endregion

#region Flatten Filter Jobs
    // Run over the For-Each index of the nativeStream counting elements
    // Put the cumulative element counts into an array
    [BurstCompile(OptimizeFor = OptimizeFor.Performance)]
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
    }

    // Counts the elements in NativeStream at job-time.
    // Output is a single-element NativeArray that works with IJobParallelForDefer Scheduling functions
    [BurstCompile]
    public struct StreamForEachCountJob : IJob
    {
        [ReadOnly] public NativeStream Stream;  // Input
        public NativeArray<int> ForEachCount0; // Output
        public void Execute() => ForEachCount0[0] = Stream.ForEachCount;
    }

    // Assemble an array from the NativeStream
    // Use the element counts to assemble the array in paralllel
    // Each thread handles a single foreach index of the native stream
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatPrecision = FloatPrecision.Low, DisableSafetyChecks = true)]
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

                if (SmoothingData.HasComponent(entityA) && SmoothingData.HasComponent(entityB))
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
    [BurstCompile(OptimizeFor = OptimizeFor.Performance)]
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
            while (reader.RemainingItemCount > 0)
            {
                DispatchPair pair = DispatchPair.CreateCollisionPair(reader.Read<BodyIndexPair>());
                UnsortedPairs[offset] = pair;
                offset++;
            }
            reader.EndForEachIndex();
        }
    }
#endregion

}