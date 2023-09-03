using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UISlider : UIComponent
    {
        private bool mDerived = false;

        public UISlider() : this(Interop.UISlider_Create(), false) { }
        public UISlider(IntPtr aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }

        ~UISlider() { if (!mDerived) Interop.UISlider_Destroy(mInstance); }
    }
}
