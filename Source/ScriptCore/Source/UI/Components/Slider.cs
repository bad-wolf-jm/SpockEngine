using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UISlider : UIComponent
    {
        private bool mDerived = false;

        public UISlider() : this(Interop.UISlider_Create(), false) { }
        public UISlider(ulong aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }

        ~UISlider() { if (!mDerived) Interop.UISlider_Destroy(mInstance); }


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong Interop.UISlider_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void Interop.UISlider_Destroy(ulong aInstance);
    }
}
