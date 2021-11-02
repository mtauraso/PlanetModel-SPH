using Unity.Entities;
using Unity.Physics.Systems;
using Unity.Mathematics;
using Unity.Jobs;
using Unity.Burst;


// Calculate pressure equation of state and 
// pressure gradient
//


[BurstCompile]
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(StepPhysicsWorld))]
[UpdateAfter(typeof(DensityFieldSystem))]
public class PressureFieldSystem : SystemBase
{
    protected override void OnUpdate()
    {
        // We depend on kernels and densities being calculated ahead of us
        var kernelSystem = World.GetExistingSystem<KernelSystem>();
        Dependency = JobHandle.CombineDependencies(kernelSystem.GetOutputDependency(), Dependency);

        var densitySystem = World.GetExistingSystem<DensityFieldSystem>();
        Dependency = JobHandle.CombineDependencies(densitySystem.GetOutputDependency(), Dependency);

        // First we will need to update all the pressures from density
        // our EOS is currently P = K(rho)^2 an approximation of liquefied hydrogen.
        // K = 1 for now, but a physically realistic value is K = 2.6*10^12 dyne*cm^4/g^2
        
        Entities.ForEach((ref ParticlePressure pressure_i, in ParticleDensity density_i) => {
            float K = 1.0f;
            float pressure = K * density_i.Value * density_i.Value;
            pressure_i.Value = pressure; 
        }).ScheduleParallel();

        // Then we will update all the pressure derivatives using the pressures as input via the
        // usual SPH formula

        // These are all the pre-calculated data we will need about the jth particle
        var interactionPairs = KernelSystem.interactionPairs;
        var kernelContributions = KernelSystem.kernelContributions;
        var massData = GetComponentDataFromEntity<ParticleMass>(true);
        var densityData = GetComponentDataFromEntity<ParticleDensity>(true);
        var pressureData = GetComponentDataFromEntity<ParticlePressure>(true);

        // Update densities particle-by-particle
        Entities.WithReadOnly(interactionPairs).WithReadOnly(kernelContributions)
            .WithReadOnly(massData).WithReadOnly(densityData).WithReadOnly(pressureData)
            .ForEach((
                Entity i,
                ref ParticlePressureGrad pressureGrad_i, // Should be write-only
                in ParticleDensity density_i, 
                in ParticleMass mass_i,
                in ParticleSmoothing smoothing_i) =>
            {
                // Self contribution to pressure Gradient?
                float3 pressure_grad = float3.zero;

                // Other particle contributions to density
                foreach (var j in interactionPairs.GetValuesForKey(i))
                {
                    var pair = new EntityOrderedPair(i, j);
                    float3 kernelGrad = kernelContributions[pair].xyz;
                    var m_j = massData[j].Value;
                    var rho_j = densityData[j].Value;
                    var P_j = pressureData[j].Value;

                    pressure_grad += kernelGrad * (m_j / rho_j * P_j);
                }

                // Pack gradient back into component.
                pressureGrad_i.Value = pressure_grad;
            }).ScheduleParallel();

        kernelSystem.AddInputDependency(Dependency);
        densitySystem.AddInputDependency(Dependency);
    }
}