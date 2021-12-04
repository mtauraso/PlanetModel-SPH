//#define DISABLE_VELOCITY_SYSTEM
#if !DISABLE_VELOCITY_SYSTEM
using Unity.Entities;
using Unity.Physics;
using Unity.Physics.Systems;
using Unity.Mathematics;
using Unity.Jobs;
using Unity.Burst;


// Run after Physics has done time integration, or rather right before physics starts
// This means the SPH calculations are always 1 timestep ahead the velocity
// integration physics is doing, so all the particle positions are one timestep behind
[BurstCompile]
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateBefore(typeof(BuildPhysicsWorld))]
public class VelocitySystem : SystemBase
{
    protected override void OnUpdate()
    {
        // We have a weird relationship with the gravity field system.
        // This is because GPU access can only occur on the main thread.
        // Since we happen First in the frame and update the velocities for
        // the upcoming frame, we call a method on gravity which must run on 
        // the main thread to pull the gravity field calculations over to CPU-land
        // This is incidentally so we can use them.
        //
        // Gravity system blocks the main thread while this happens
        var gravityFieldSystem = World.GetExistingSystem<GravityFieldSystem>();
        //gravityFieldSystem.EnsureFieldComputationComplete();

        // Need to get dt for velocity updates
        var dt = World.Time.DeltaTime;

        // Schedule a job to apply gravity & Pressure to determine particle velocity.
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
