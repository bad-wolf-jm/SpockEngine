using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIColorButton : UIComponent
    {
        private bool mDerived = false;

        public UIColorButton() : this(UIColorButton_Create(), false) { }
        public UIColorButton(ulong aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }

        ~UIColorButton() { if (!mDerived) UIColorButton_Destroy(mInstance); }
    }
}
