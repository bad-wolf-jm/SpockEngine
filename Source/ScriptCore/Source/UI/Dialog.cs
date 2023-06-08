using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIDialog : UIComponent
    {
        bool mDerived = false;
        public UIDialog() : this(UIDialog_Create(), false) { }

        public UIDialog(ulong aInstance, bool aDerived) : base(aInstance) { mDerived = aDerived; }

        ~UIDialog() { UIDialog_Destroy(mInstance); }

        public void SetTitle(string aTitle) { UIDialog_SetTitle(mInstance, aTitle); }

        public void SetContent(UIComponent aContent) { UIDialog_SetContent(mInstance, aContent.Instance); }

        public void SetSize(Math.vec2 aSize) { UIDialog_SetSize(mInstance, aSize); }

        public void Update() { UIDialog_Update(mInstance); }
        
        public void Open() { UIDialog_Open(mInstance); }
        
        public void Close() { UIDialog_Close(mInstance); }
    }
}
