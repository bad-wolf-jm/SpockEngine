using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UITextOverlay : UIComponent
    {

        public UITextOverlay() : base(UITextOverlay_Create()) { }
        public UITextOverlay(ulong aSelf) : base(aSelf) { }

        ~UITextOverlay() { UITextOverlay_Destroy(mInstance); }

        public void AddText(string aText) { UITextOverlay_AddText(mInstance, aText); }
        public void Clear() { UITextOverlay_Clear(mInstance); }
    }
}
