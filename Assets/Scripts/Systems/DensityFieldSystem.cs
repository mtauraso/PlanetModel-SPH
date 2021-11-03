using Unity.Entities;
using Unity.Physics.Systems;
using Unity.Jobs;
using Unity.Burst;


[BurstCompile]
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(KernelDataSystem))]
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

#if !KERNEL_DYNAMIC_BUFFER
        // Get the interaciton pairs and kernel values for those pairs populated by the KernelSystem
        var interactionPairs = KernelSystem.interactionPairs;
        var kernelContributions = KernelSystem.kernelContributions;
#endif
        // RO access to particle mass data
        var massData = GetComponentDataFromEntity<ParticleMass>(true); 
        
       
        // Update densities particle-by-particle
        Entities.WithReadOnly(massData)
#if !KERNEL_DYNAMIC_BUFFER
            .WithReadOnly(interactionPairs).WithReadOnly(kernelContributions)
            
#endif
            .ForEach(( 
                Entity i, 
                ref ParticleDensity density_i, // Should be write-only
#if KERNEL_DYNAMIC_BUFFER
                in DynamicBuffer<ParticleInteraction> interactions,
#endif
                in ParticleMass mass_i, 
                in ParticleSmoothing smoothing_i) =>
        {
            // Self contribution to density
            var density = mass_i.Value * SplineKernel.Kernel(0, smoothing_i.h);

            // Other particle contributions to density
#if KERNEL_DYNAMIC_BUFFER
            foreach(ParticleInteraction interaction in interactions)
            {
                Entity j = interaction.Other;
                var kernel = interaction.Kernel.w;
                density += massData[j].Value * kernel;
            }
#else
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
#endif
            
            density_i.Value = density;
        }).ScheduleParallel();

        kernelSystem.AddInputDependency(Dependency);
    }
}


