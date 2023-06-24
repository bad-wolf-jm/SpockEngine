using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UICodeEditor : UIComponent
    {
        private bool mDerived = false;

        public UICodeEditor() : this(Interop.UICodeEditor_Create(), false) { }
        public UICodeEditor(IntPtr aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }
        ~UICodeEditor() { if (!mDerived) Interop.UICodeEditor_Destroy(mInstance); }

        public void SetText(string aText) { Interop.UICodeEditor_SetText(mInstance, aText); }
        public string GetText(string aText) { return Interop.UICodeEditor_GetText(mInstance); }
    }
}
