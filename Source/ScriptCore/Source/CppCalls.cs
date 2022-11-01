using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public static class CppCall
    {
        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static void NativeLog(string aString, int i);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static bool Entity_IsValid(uint aEntityID, ulong aRegistry);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static bool Entity_Has(uint aEntityID, ulong aRegistry, Type aTypeDesc);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        internal extern static T Entity_Get<T>(uint aEntityID, ulong aRegistry, Type aTypeDesc) where T : Component, new();

        // [MethodImplAttribute(MethodImplOptions.InternalCall)]
        // internal extern static object GetScriptInstance(ulong entityID);

        // [MethodImplAttribute(MethodImplOptions.InternalCall)]
        // internal extern static void TransformComponent_GetTranslation(ulong entityID, out Vector3 translation);

        // [MethodImplAttribute(MethodImplOptions.InternalCall)]
        // internal extern static void TransformComponent_SetTranslation(ulong entityID, ref Vector3 translation);

        // [MethodImplAttribute(MethodImplOptions.InternalCall)]
        // internal extern static void Rigidbody2DComponent_ApplyLinearImpulse(ulong entityID, ref Vector2 impulse, ref Vector2 point, bool wake);

        // [MethodImplAttribute(MethodImplOptions.InternalCall)]
        // internal extern static void Rigidbody2DComponent_ApplyLinearImpulseToCenter(ulong entityID, ref Vector2 impulse, bool wake);

        // [MethodImplAttribute(MethodImplOptions.InternalCall)]
        // internal extern static bool Input_IsKeyDown(KeyCode keycode);
    }
}