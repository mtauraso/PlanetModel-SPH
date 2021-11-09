#define LIST_OF_QUEUES
//#define COMMAND_BUFFER
//#define RECORD_ALL_COLLISIONS 
// Debug flag to help understand what is being detected as
// a collision in simple situations, verify correctness

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

#if RECORD_ALL_COLLISIONS
[InternalBufferCapacity(8)]
public struct ParticleCollision : IBufferElementData
{
    public Entity Other;
    public float distance;
}
#endif

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

[BurstCompile]
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(BuildPhysicsWorld))]
[UpdateBefore(typeof(StepPhysicsWorld))]
public class KernelSystem : SystemBase, IParticleSystem
{
    private StepPhysicsWorld m_StepPhysicsWorld;

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

        // Reset our input Dependency, so we can start collecting for the next frame
        inputDependency = default;
    }

    protected override void OnUpdate()
    {
        Cleanup();

        m_StepPhysicsWorld.EnqueueCallback(SimulationCallbacks.Phase.PostCreateDispatchPairs,
           (ref ISimulation sim, ref PhysicsWorld pw, JobHandle inputDeps) =>
           { 
                int batchCount = 1; // How many interaction pairs should a worker process before returning to the pool?



               // Job to create an appropriately sized list of NativeQueues

               // Hijack the indexing scheme used in physics for ordering rigid bodies
               // This is a strict over-count of particles but not by much
               int numParticles = pw.Bodies.Length;

               NativeStream particleInteractions = new NativeStream(numParticles, Allocator.TempJob);

               // Job to evaluate kernel across and push into native queues
               // Try Get sim.StepContext.PhasedDispatchPais with reflection
               // This is an internal array in the physics system of object pairs from the broadphase.
               // We need this array to distribute processing of pairs across threads
               // What we are doing is essentially an IBodyPairsJob, but multithreaded
               // It is too bad there is no interface for this in Physics
               var stepContextFieldInfo = sim.GetType().GetField("StepContext", BindingFlags.NonPublic | BindingFlags.Instance);
               var stepContext = stepContextFieldInfo.GetValue(sim);
               var phasedDispatchPairsFieldInfo = stepContext.GetType().GetField("PhasedDispatchPairs", BindingFlags.Public | BindingFlags.Instance);
               var phasedDispatchPairs = (NativeList<DispatchPairSequencer.DispatchPair>)phasedDispatchPairsFieldInfo.GetValue(stepContext);

               // Also need the solver scheduler info so we can iterate over body pairs multithreaded (similar to solver)
               var solverSchedulerInfoFieldInfo = stepContext.GetType().GetField("SolverSchedulerInfo", BindingFlags.Public | BindingFlags.Instance);
               var SolverSchedulerInfo = typeof(DispatchPairSequencer).Assembly.GetType("SolverSchedulerInfo");
               var solverSchedulerInfo = solverSchedulerInfoFieldInfo.GetValue(stepContext);
               Type.DefaultBinder.ChangeType(solverSchedulerInfo, SolverSchedulerInfo, System.Globalization.CultureInfo.CurrentCulture);


               var capturePairsTest = new InteractionPairsJob()
               {
                   phasedDispatchPairs = phasedDispatchPairs,
                   bodies = pw.Bodies,
                   smoothingData = GetComponentDataFromEntity<ParticleSmoothing>(true), // RO access to particle smoothing size
                   translationData = GetComponentDataFromEntity<Translation>(true),
                   particleInteractionsWriter = particleInteractions.AsWriter(),
                };
                inputDeps = capturePairsTest.Schedule(phasedDispatchPairs, batchCount, inputDeps);

               // Job to disable further physics calculation on our pairs
                var disablePairs = new DisablePairsJob();
                inputDeps = disablePairs.Schedule(sim, ref pw, inputDeps);

               // Job to go over our particles and copy the appropriate queue of interactions into
               // the dynamic buffers
               var copyInteractionJob = new CopyInteractionJob()
               {
                   bodies = pw.Bodies,
                   particleInteractionsReader = particleInteractions.AsReader(),
                   bufferLookup = GetBufferFromEntity<ParticleInteraction>(),
               };
               inputDeps = copyInteractionJob.Schedule(numParticles, 1, inputDeps);

               // Job to dispose all our queues
               var disposeQueuesJob = new DisposeQueuesJob()
               {
                   particleInteractions = particleInteractions,
               };
               inputDeps = disposeQueuesJob.Schedule(inputDeps);

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
            // only step-velocity integration!
        }
    }

    public struct CopyInteractionJob : IJobParallelFor
    {
        [ReadOnly]
        public NativeArray<RigidBody> bodies;

        [WriteOnly]
        [NativeDisableParallelForRestriction]
        public BufferFromEntity<ParticleInteraction> bufferLookup;

        [ReadOnly]
        public NativeStream.Reader particleInteractionsReader;

        public void Execute(int index)
        {
            Entity e = bodies[index].Entity;

            if (bufferLookup.HasComponent(e))
            {
                DynamicBuffer<ParticleInteraction> buffer = bufferLookup[e];

                particleInteractionsReader.BeginForEachIndex(index);
                
                while ( particleInteractionsReader.RemainingItemCount > 0)
                {
                    ParticleInteraction element = particleInteractionsReader.Read<ParticleInteraction>();
                    buffer.Add(element);
                }

                particleInteractionsReader.EndForEachIndex();
            }
        }
    }

    public struct DisposeQueuesJob : IJob
    {
        public NativeStream particleInteractions;

        public void Execute()
        {
            particleInteractions.Dispose();
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
        [NativeDisableParallelForRestriction]
        public NativeStream.Writer particleInteractionsWriter;


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
            var ij = new ParticleInteraction {
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
                particleInteractionsWriter.BeginForEachIndex(dispatchPair.BodyIndexA);
                particleInteractionsWriter.Write<ParticleInteraction>(ij);
                particleInteractionsWriter.EndForEachIndex();
                
                // Particle j
                particleInteractionsWriter.BeginForEachIndex(dispatchPair.BodyIndexB);
                particleInteractionsWriter.Write<ParticleInteraction>(ji);
                particleInteractionsWriter.EndForEachIndex();
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