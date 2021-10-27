using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;


// Thread safe RNG for parallel jobs. Intended to be Burst compatible
// Following approach of : https://reeseschultz.com/random-number-generation-with-unity-dots/
// 
// Adding in a factory and Random wrapper object so that the need to "put back" the RNG state in the global array
// is explicitly a responsibility of the caller.
//
// TODO: Fix errors to make this burst compatible
//
// TODO: Using a System as the global point of access is a little awkward, can this simply be a static class
//       with just-in-time initialization when someone calls getRngFactory? If so, how would destruction or multiple
//       ECS worlds be handled (if at all)

[UpdateInGroup(typeof(InitializationSystemGroup))]
public class RandomSystem : ComponentSystem
{
    public NativeArray<Unity.Mathematics.Random> RandomArray { get; private set; }

    public RngFactory getRngFactory()
    {
        return new RngFactory
        {
            RandomArray = RandomArray
        };
     }

    protected override void OnCreate()
    {
        var randomArray = new Unity.Mathematics.Random[JobsUtility.MaxJobThreadCount];
        var seed = new System.Random();

        for (var i = 0; i < JobsUtility.MaxJobThreadCount; ++i)
            randomArray[i] = new Unity.Mathematics.Random((uint)seed.Next());

        RandomArray = new NativeArray<Unity.Mathematics.Random>(randomArray, Allocator.Persistent);
    }

    protected override void OnDestroy()
        => RandomArray.Dispose();

    protected override void OnUpdate() { }
}

// TODO: Need better names for these interface objects.
//
// This objects lifecycle is that of a job, you should be able to place it on a job
// It'll get re-created as a value type for each thread in the job.
//
// It needs to be a value type to work with Unity's job system, so you have to invoke
// A function on it once you're in the thread to get something which acts like an RNG,
// See below.
//
// Part of the idea here is we wrap all the scary thread unsafe bits, and also align object
// lifecycles with the necessary updates, so someone using the class simply
// gets the appropriate thing for the part of the job system they're in and then uses it
// and the code handles the needed cleanup behind the scenes.
    
public struct RngFactory
{
    [NativeDisableParallelForRestriction]
    public NativeArray<Unity.Mathematics.Random> RandomArray;

    [NativeSetThreadIndex]
    private int nativeThreadId;

    public Random GetRng()
    {
        return new Random
        {
            _rng = RandomArray[nativeThreadId],
            _rngFactory = this,
        };
    }

    // This is a class which acts essentially as a reference to a thread-specific
    // Unity.Mathematics.Random RNG state. It implements IDisposable because the caller
    // must dispose of it when the thread is done with it.
    //
    // calling Dispose (explicity or via using..) allows us to put the state back into the
    // Global array of per-thread RNG states in RandomSystem, and ensures that we do not get
    // repeated sequences of random numbers on the same thread.
    //
    // This object passes through calls to Unity.Mathematics.Random's interface, it has to because
    // This is the only way to keep the per-thread rng state updated after each call on a particular thread.
    // It would be more direct to hand out a reference to the Unity.Mathematics.Random object in the global array;
    // however this is not possible without using C# unsafe.
    public class Random : IDisposable
    {
        public Unity.Mathematics.Random _rng;
        public RngFactory _rngFactory;

        public void Dispose()
        {
            _rngFactory.RandomArray[_rngFactory.nativeThreadId] = _rng;
        }

        // This is a little silly, but Unity.Mathematics.Random is sealed!
        // Ideally we would inherit to get pass-through of methods, but we must use composition
        //
        // Our alternative to the approach below is reflection. Note that:
        // 1) Reflection is not recommended for use inside Unity because it creates
        //    many objects which do not get garbage collected.
        // 2) Inside the inner loop of a job (where we intend this to run) we hope to have
        //    these method calls inlined
        //
        //    Note: I have not checked whether C# or burst is inlining these in their current
        //    form; however, this approach or similar has the best chance to be inlined
        //    automatically
        //
        // Consider also:
        // 1) this interface is also unlikely to change significantly and is
        //    easy to update to match Unity.Mathematics.Random. Just copy paste!
        // 2) Autocomplete works on these objects in Visual Studio without any additional
        //    hackery, which would not be true if we had used reflection.
        //
        public void InitState(uint seed = 1851936439) => _rng.InitState(seed);
        public bool NextBool() => _rng.NextBool();
        public bool2 NextBool2() => _rng.NextBool2();
        public bool3 NextBool3() => _rng.NextBool3();
        public bool4 NextBool4() => _rng.NextBool4();
        public double NextDouble(double min, double max) => _rng.NextDouble(min, max);
        public double NextDouble(double max) => _rng.NextDouble(max);
        public double NextDouble() => _rng.NextDouble();
        public double2 NextDouble2(double2 max) => _rng.NextDouble2(max);
        public double2 NextDouble2() => _rng.NextDouble2();
        public double2 NextDouble2(double2 min, double2 max) => _rng.NextDouble2(min, max);
        public double2 NextDouble2Direction() => _rng.NextDouble2Direction();
        public double3 NextDouble3(double3 max) => _rng.NextDouble3(max);
        public double3 NextDouble3(double3 min, double3 max) => _rng.NextDouble3(min, max);
        public double3 NextDouble3() => _rng.NextDouble3();
        public double3 NextDouble3Direction() => _rng.NextDouble3Direction();
        public double4 NextDouble4(double4 max) => _rng.NextDouble4(max);
        public double4 NextDouble4(double4 min, double4 max) => _rng.NextDouble4(min, max);
        public double4 NextDouble4() => _rng.NextDouble4();
        public float NextFloat(float min, float max) => _rng.NextFloat(min, max);
        public float NextFloat(float max) => _rng.NextFloat(max);
        public float NextFloat() => _rng.NextFloat();
        public float2 NextFloat2(float2 min, float2 max) => _rng.NextFloat2(min, max);
        public float2 NextFloat2(float2 max) => _rng.NextFloat2(max);
        public float2 NextFloat2() => _rng.NextFloat2();
        public float2 NextFloat2Direction() => _rng.NextFloat2Direction();
        public float3 NextFloat3(float3 min, float3 max) => _rng.NextFloat3(min, max);
        public float3 NextFloat3(float3 max) => _rng.NextFloat3(max);
        public float3 NextFloat3() => _rng.NextFloat3();
        public float3 NextFloat3Direction() => _rng.NextFloat3Direction();
        public float4 NextFloat4() => _rng.NextFloat4();
        public float4 NextFloat4(float4 max) => _rng.NextFloat4(max);
        public float4 NextFloat4(float4 min, float4 max) => _rng.NextFloat4(min, max);
        public int NextInt(int max) => _rng.NextInt(max);
        public int NextInt(int min, int max) => _rng.NextInt(min, max);
        public int NextInt() => _rng.NextInt();
        public int2 NextInt2(int2 max) => _rng.NextInt2(max);
        public int2 NextInt2(int2 min, int2 max) => _rng.NextInt2(min, max);
        public int2 NextInt2() => _rng.NextInt2();
        public int3 NextInt3(int3 max) => _rng.NextInt3(max);
        public int3 NextInt3(int3 min, int3 max) => _rng.NextInt3(min, max);
        public int3 NextInt3() => _rng.NextInt3();
        public int4 NextInt4() => _rng.NextInt4();
        public int4 NextInt4(int4 max) => _rng.NextInt4(max);
        public int4 NextInt4(int4 min, int4 max) => _rng.NextInt4(min, max);
        public quaternion NextQuaternionRotation() => _rng.NextQuaternionRotation();
        public uint NextUInt(uint min, uint max) => _rng.NextUInt(min, max);
        public uint NextUInt(uint max) => _rng.NextUInt(max);
        public uint NextUInt() => _rng.NextUInt();
        public uint2 NextUInt2(uint2 min, uint2 max) => _rng.NextUInt2(min, max);
        public uint2 NextUInt2(uint2 max) => _rng.NextUInt2(max);
        public uint2 NextUInt2() => _rng.NextUInt2();
        public uint3 NextUInt3(uint3 max) => _rng.NextUInt3(max);
        public uint3 NextUInt3() => _rng.NextUInt3();
        public uint3 NextUInt3(uint3 min, uint3 max) => _rng.NextUInt3(min, max);
        public uint4 NextUInt4(uint4 max) => _rng.NextUInt4(max);
        public uint4 NextUInt4() => _rng.NextUInt4();
        public uint4 NextUInt4(uint4 min, uint4 max) => _rng.NextUInt4(min, max);
    }
}
 