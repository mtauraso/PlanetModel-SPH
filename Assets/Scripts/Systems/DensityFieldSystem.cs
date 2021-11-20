using Unity.Entities;
using Unity.Physics.Systems;
using Unity.Jobs;
using Unity.Burst;


[BurstCompile]
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(StepPhysicsWorld))]
public class DensityFieldSystem : SystemBase, IPhysicsSystem
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

        // NOTE: We should not need this dependency. It arises because both us and gravity field system
        //       use mass data. I think the job system can't tell that everyone is only reading mass data
        //       and that mass data is never written on a per-frame basis.
        var gravityFieldSystem = World.GetExistingSystem<GravityFieldSystem>();
        Dependency = JobHandle.CombineDependencies(gravityFieldSystem.GetOutputDependency(), Dependency);

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


