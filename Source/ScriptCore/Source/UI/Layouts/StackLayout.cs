using System;
using System.Runtime.CompilerServices;

using SpockEngine.Math;

namespace SpockEngine
{
    public class StackLayout : UIComponent
    {
        public StackLayout() : base(StackLayout_Create()) { }

        ~StackLayout() { StackLayout_Destroy(mInstance); }

        public void Add(UIComponent aChild, string aKey)
        {
            StackLayout_Add(mInstance, aChild.Instance, aKey);
        }

        public void SetCurrent(string aKey)
        {
            StackLayout_SetCurrent(mInstance, aKey);
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong StackLayout_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void StackLayout_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void StackLayout_Add(ulong aInstance, ulong aChild, string aKey);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void StackLayout_SetCurrent(ulong aInstance, string aKey);
    }
}