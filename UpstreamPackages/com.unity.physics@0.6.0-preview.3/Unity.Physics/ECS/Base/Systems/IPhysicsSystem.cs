using Unity.Jobs;

namespace Unity.Physics.Systems
{
    // Defines an interface for all internal physics systems
    public interface IPhysicsSystem
    {
        void AddInputDependency(JobHandle inputDep);

        JobHandle GetOutputDependency();
    }
}
