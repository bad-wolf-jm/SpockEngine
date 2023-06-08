using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UITextOverlay : UIComponent
    {

        public UITextOverlay() : base(Interop.UITextOverlay_Create()) { }
        public UITextOverlay(IntPtr aSelf) : base(aSelf) { }

        ~UITextOverlay() { Interop.UITextOverlay_Destroy(mInstance); }

        public void AddText(string aText) { Interop.UITextOverlay_AddText(mInstance, aText); }
        public void Clear() { Interop.UITextOverlay_Clear(mInstance); }
    }
}
