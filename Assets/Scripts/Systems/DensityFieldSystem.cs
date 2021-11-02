using Unity.Entities;
using Unity.Physics.Systems;
using Unity.Jobs;
using Unity.Burst;


[BurstCompile]
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(StepPhysicsWorld))]
public class DensityFieldSystem : SystemBase, IParticleSystem
{
    public void AddInputDependency(JobHandle jh)
    {
        inputDependency = JobHandle.CombineDependencies(jh, inputDependency);
    }
    private JobHandle inputDependency;
    public JobHandle GetOutputDependency() => Dependency;

    protected override void OnUpdate()
    {
        // ensure any system that registered a dependency is done with our data
        inputDependency.Complete();

        //Debug.Log("DensityFieldSystem on Update");
        var kernelSystem = World.GetExistingSystem<KernelSystem>();
        Dependency = JobHandle.CombineDependencies(kernelSystem.GetOutputDependency(), Dependency);

        // Get the interaciton pairs and kernel values for those pairs populated by the KernelSystem
        var interactionPairs = KernelSystem.interactionPairs;
        var kernelContributions = KernelSystem.kernelContributions;
        // RO access to particle mass data
        var massData = GetComponentDataFromEntity<ParticleMass>(true); 
        
        // Update densities particle-by-particle
        Entities
            .WithReadOnly(interactionPairs)
            .WithReadOnly(kernelContributions)
            .WithReadOnly(massData)
            .ForEach(( 
                Entity i, 
                ref ParticleDensity density_i, // Should be write-only
                in ParticleMass mass_i, 
                in ParticleSmoothing smoothing_i) =>
        {
            // Self contribution to density
            var density = mass_i.Value * SplineKernel.Kernel(0, smoothing_i.h);
            
            // Other particle contributions to density
            
            foreach(var j in interactionPairs.GetValuesForKey(i))
            {
                var pair = new EntityOrderedPair(i, j);
#if USE_KC
                var kernel = kernelContributions[pair].kernel();
#else
                var kernel = kernelContributions[pair].w;
#endif
                density += massData[j].Value * kernel;
            }
            
            density_i.Value = density;
        }).ScheduleParallel();

        kernelSystem.AddInputDependency(Dependency);
    }
}


