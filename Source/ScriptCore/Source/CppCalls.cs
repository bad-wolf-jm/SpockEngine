using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public static class CppCall
    {
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint Entity_Create(ulong aRegistry, string aName, uint aParentEntityID);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static bool Entity_IsValid(uint aEntityID, ulong aRegistry);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static bool Entity_Has(uint aEntityID, ulong aRegistry, Type aTypeDesc);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static T Entity_Get<T>(uint aEntityID, ulong aRegistry, Type aTypeDesc) where T : Component, new();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void Entity_Add<T>(uint aEntityID, ulong aRegistry, Type aTypeDesc, T aNewValue) where T : Component, new();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void Entity_Replace<T>(uint aEntityID, ulong aRegistry, Type aTypeDesc, T aNewValue) where T : Component, new();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void Entity_Remove(uint aEntityID, ulong aRegistry, Type aTypeDesc);


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static ulong OpNode_NewTensorShape( uint[,] aShape, uint aRank, uint aLayers, uint aElementSize );

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static ulong OpNode_DestroyTensorShape( ulong aTensorShape );

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint OpNode_CountLayers( ulong aTensorShape );

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static uint[] OpNode_GetDimension( ulong aTensorShape, int i );

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void OpNode_Trim( ulong aTensorShape, int aToDimension );

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void OpNode_Flatten( ulong aTensorShape, int aToDimension );

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void OpNode_InsertDimension( ulong aTensorShape, int aPosition, uint[] aDimension );

    }
}