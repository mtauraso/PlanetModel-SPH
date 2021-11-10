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
        updateNumber++;
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