using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIMarkdown : UIComponent
    {
        private bool mDerived = false;
        public UIMarkdown() : this(Interop.UIMarkdown_Create(), false) { }
        public UIMarkdown(IntPtr aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }

        ~UIMarkdown() { if (!mDerived) Interop.UIMarkdown_Destroy(mInstance); }

        public void SetText(string aText) { Interop.UIMarkdown_SetText(mInstance, aText); }

        public void SetTextColor(Math.vec4 aColor) { Interop.UIMarkdown_SetTextColor(mInstance, aColor); }
    }
}
