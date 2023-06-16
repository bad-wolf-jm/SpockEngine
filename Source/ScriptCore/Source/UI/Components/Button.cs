using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public class UIButton : UIComponent
    {
        public UIButton() : base(Interop.UIButton_Create()) { }
        public UIButton(string aText) : this()
        {
            SetText(aText);
        }

        ~UIButton() { Interop.UIButton_Destroy(mInstance); }

        public void SetText(string aText) { Interop.UIButton_SetText(mInstance, aText); }

        public delegate void ClickDelegate();
        ClickDelegate onClick;
        public void OnClick(ClickDelegate aHandler)
        {
            onClick = aHandler;

            Interop.UIButton_OnClick(mInstance, Marshal.GetFunctionPointerForDelegate(onClick));
        }
    }
}
