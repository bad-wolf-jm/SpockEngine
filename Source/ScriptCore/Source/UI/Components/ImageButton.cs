using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public class UIImageButton : UIBaseImage
    {
        public UIImageButton() : base(Interop.UIImageButton_Create(), true) { }
        public UIImageButton(string aText, Math.vec2 aSize) : base(Interop.UIImageButton_CreateWithPath(aText, aSize), true) { }

        ~UIImageButton() { Interop.UIImageButton_Destroy(mInstance); }


        public delegate void ClickDelegate();
        ClickDelegate onClick;
        public void OnClick(ClickDelegate aHandler)
        {
            onClick = aHandler;

            Interop.UIImageButton_OnClick(mInstance,  Marshal.GetFunctionPointerForDelegate(onClick));
        }
    }
}
