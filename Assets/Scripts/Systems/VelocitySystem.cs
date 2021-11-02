#define DISABLE_VELOCITY_SYSTEM
#if !DISABLE_VELOCITY_SYSTEM
using Unity.Entities;
using Unity.Physics;
using Unity.Physics.Systems;
using Unity.Mathematics;
using Unity.Jobs;
using Unity.Burst;


// Run after Physics has done time integration
// This means the SPH calculations are always 1 timestep ahead the velocity
// integration physics is doing, so all the particle positions are one timestep behind
[BurstCompile]
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(ExportPhysicsWorld))]
public class VelocitySystem : SystemBase
{
    protected override void OnUpdate()
    {
        // Need to get dt for velocity updates
        var dt = World.Time.DeltaTime;

        Entities.ForEach((int entityInQueryIndex, Entity i,
            ref PhysicsVelocity vel, in ParticleDensity density_i, in ParticlePressureGrad pressureGrad_i, in GravityField gravity_i) =>
        {
            // First need to EOM ourselves from Pressure, density, gravity, inertia to dv/dt
            float3 GradP = pressureGrad_i.Value;
            float3 GradPhi = gravity_i.Value.xyz;
            float rho = density_i.Value;

            float3 dvdt = -GradP / rho - GradPhi;
            // Then we need to update v based on dv/dt and dt
            vel.Linear = vel.Linear + dvdt * dt;

        }).ScheduleParallel();
    }
}
#endif
