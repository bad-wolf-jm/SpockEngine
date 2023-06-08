using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public class UIButton : UIComponent
    {
        public UIButton() : base(UIButton_Create()) { }
        public UIButton(string aText) : base(UIButton_CreateWithText(aText)) { }

        ~UIButton() { UIButton_Destroy(mInstance); }

        public void SetText(string aText) { UIButton_SetText(mInstance, aText); }

        public delegate void ClickDelegate();
        ClickDelegate onClick;
        public void OnClick(ClickDelegate aHandler)
        {
            onClick = aHandler;

            UIButton_OnClick(mInstance, Marshal.GetFunctionPointerForDelegate(onClick));
        }
    }
}
