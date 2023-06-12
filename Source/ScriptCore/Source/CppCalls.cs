using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public static class CppCall
    {
        [DllImport("LTSimulationEngineRuntime.dll", CharSet = CharSet.Unicode)]
        [return: MarshalAs(UnmanagedType.LPWStr)]
        public extern static string OpenFile(string aFilter);

        // [MethodImplAttribute(MethodImplOptions.InternalCall)]
        // internal extern static uint Entity_Create(ulong aRegistry, string aName, uint aParentEntityID);

        // [MethodImplAttribute(MethodImplOptions.InternalCall)]
        // internal extern static bool Entity_IsValid(uint aEntityID, ulong aRegistry);

        // [MethodImplAttribute(MethodImplOptions.InternalCall)]
        // internal extern static bool Entity_Has(uint aEntityID, ulong aRegistry, Type aTypeDesc);

        // [MethodImplAttribute(MethodImplOptions.InternalCall)]
        // internal extern static T Entity_Get<T>(uint aEntityID, ulong aRegistry, Type aTypeDesc) where T : Component, new();

        // [MethodImplAttribute(MethodImplOptions.InternalCall)]
        // internal extern static void Entity_Add<T>(uint aEntityID, ulong aRegistry, Type aTypeDesc, T aNewValue) where T : Component, new();

        // [MethodImplAttribute(MethodImplOptions.InternalCall)]
        // internal extern static void Entity_Replace<T>(uint aEntityID, ulong aRegistry, Type aTypeDesc, T aNewValue) where T : Component, new();

        // [MethodImplAttribute(MethodImplOptions.InternalCall)]
        // internal extern static void Entity_Remove(uint aEntityID, ulong aRegistry, Type aTypeDesc);
    }
}