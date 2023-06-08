using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UISplitter : UIComponent
    {
        private bool mDerived = false;
        public UISplitter() : this(UISplitter_Create(), false) { }

        public UISplitter(ulong aInstance, bool aDerived) : base(aInstance) { mDerived = aDerived; }

        public UISplitter(eBoxLayoutOrientation aOrientation) : this(UISplitter_CreateWithOrientation(aOrientation), false) { }

        ~UISplitter() { if (!mDerived) UISplitter_Destroy(mInstance); }

        public void SetItemSpacing(float aItemSpacing)
        {
            UISplitter_SetItemSpacing(mInstance, aItemSpacing);
        }

        public void Add1(UIComponent aChild)
        {
            var lInstance = (aChild != null) ? aChild.Instance : 0;

            UISplitter_Add1(mInstance, lInstance);
        }

        public void Add2(UIComponent aChild)
        {
            var lInstance = (aChild != null) ? aChild.Instance : 0;

            UISplitter_Add2(mInstance, lInstance);
        }
    }
}