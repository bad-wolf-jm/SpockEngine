using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public class UIImageButton : UIBaseImage
    {
        public UIImageButton() : base(UIImageButton_Create(), true) { }
        public UIImageButton(string aText, Math.vec2 aSize) : base(UIImageButton_CreateWithPath(aText, aSize), true) { }

        ~UIImageButton() { UIImageButton_Destroy(mInstance); }


        public delegate void ClickDelegate();
        ClickDelegate onClick;
        public void OnClick(ClickDelegate aHandler)
        {
            onClick = aHandler;

            UIImageButton_OnClick(mInstance,  Marshal.GetFunctionPointerForDelegate(onClick));
        }
    }
}
