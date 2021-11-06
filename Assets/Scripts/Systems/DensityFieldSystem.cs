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

        // RO access to particle mass data
        var massData = GetComponentDataFromEntity<ParticleMass>(true); 
        
       
        // Update densities particle-by-particle
        Entities.WithReadOnly(massData).ForEach(( Entity i, 
                ref ParticleDensity density_i, // Should be write-only
                in DynamicBuffer<ParticleInteraction> interactions,
                in ParticleMass mass_i, 
                in ParticleSmoothing smoothing_i) =>
        {
            // Self contribution to density
            var density = mass_i.Value * SplineKernel.Kernel(0, smoothing_i.h);

            // Other particle contributions to density
            foreach(ParticleInteraction interaction in interactions)
            {
                Entity j = interaction.Other;
                var kernel = interaction.KernelSymmetric.w;
                density += massData[j].Value * kernel;
            }

            density_i.Value = density;
        }).ScheduleParallel();

        kernelSystem.AddInputDependency(Dependency);
    }
}


