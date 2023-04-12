using System;
using System.Runtime.CompilerServices;

using SpockEngine.Math;

namespace SpockEngine
{
    public class UIStackLayout : UIComponent
    {
        public UIStackLayout() : base(UIStackLayout_Create()) { }

        ~UIStackLayout() { UIStackLayout_Destroy(mInstance); }

        public void Add(UIComponent aChild, string aKey)
        {
            UIStackLayout_Add(mInstance, aChild.Instance, aKey);
        }

        public void SetCurrent(string aKey)
        {
            UIStackLayout_SetCurrent(mInstance, aKey);
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIStackLayout_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIStackLayout_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIStackLayout_Add(ulong aInstance, ulong aChild, string aKey);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIStackLayout_SetCurrent(ulong aInstance, string aKey);
    }
}