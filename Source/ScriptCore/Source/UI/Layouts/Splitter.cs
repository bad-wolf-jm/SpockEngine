using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UISplitter : UIComponent
    {
        private bool mDerived = false;
        public UISplitter() : this(Interop.UISplitter_Create(), false) { }

        public UISplitter(IntPtr aInstance, bool aDerived) : base(aInstance) { mDerived = aDerived; }

        public UISplitter(eBoxLayoutOrientation aOrientation) : this()
        {
            SetOrientation(aOrientation);
        }

        ~UISplitter() { if (!mDerived) Interop.UISplitter_Destroy(mInstance); }

        public void SetItemSpacing(float aItemSpacing)
        {
            Interop.UISplitter_SetItemSpacing(mInstance, aItemSpacing);
        }

        public void Add1(UIComponent aChild)
        {
            var lInstance = (aChild != null) ? aChild.Instance : IntPtr.Zero;

            Interop.UISplitter_Add1(mInstance, lInstance);
        }

        public void Add2(UIComponent aChild)
        {
            var lInstance = (aChild != null) ? aChild.Instance : IntPtr.Zero;

            Interop.UISplitter_Add2(mInstance, lInstance);
        }
    }
}