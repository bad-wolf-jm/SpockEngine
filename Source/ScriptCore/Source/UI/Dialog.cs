using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIDialog : UIComponent
    {
        bool mDerived = false;
        public UIDialog() : this(Interop.UIDialog_Create(), false) { }

        public UIDialog(IntPtr aInstance, bool aDerived) : base(aInstance) { mDerived = aDerived; }

        ~UIDialog() { Interop.UIDialog_Destroy(mInstance); }

        public void SetTitle(string aTitle) { Interop.UIDialog_SetTitle(mInstance, aTitle); }

        public void SetContent(UIComponent aContent) { Interop.UIDialog_SetContent(mInstance, aContent.Instance); }

        public void SetSize(Math.vec2 aSize) { Interop.UIDialog_SetSize(mInstance, aSize); }

        public void Update() { Interop.UIDialog_Update(mInstance); }
        
        public void Open() { Interop.UIDialog_Open(mInstance); }
        
        public void Close() { Interop.UIDialog_Close(mInstance); }
    }
}
