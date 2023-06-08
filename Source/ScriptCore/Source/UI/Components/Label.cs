using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UILabel : UIComponent
    {
        private bool mDerived = false;

        public UILabel() : this(UILabel_Create(), false) { }
        public UILabel(ulong aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }
        public UILabel(string aText) : this(UILabel_CreateWithText(aText), false) { }

        ~UILabel() { if (!mDerived) UILabel_Destroy(mInstance); }

        public void SetText(string aText) { UILabel_SetText(mInstance, aText); }

        public void SetTextColor(Math.vec4 aColor) { UILabel_SetTextColor(mInstance, aColor); }
    }
}
