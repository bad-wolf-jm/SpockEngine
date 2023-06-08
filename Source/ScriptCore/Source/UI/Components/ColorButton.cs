using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIColorButton : UIComponent
    {
        private bool mDerived = false;

        public UIColorButton() : this(Interop.UIColorButton_Create(), false) { }
        public UIColorButton(IntPtr aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }

        ~UIColorButton() { if (!mDerived) Interop.UIColorButton_Destroy(mInstance); }
    }
}
