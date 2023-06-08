using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIDropdownButton : UIComponent
    {
        public UIDropdownButton() : base(Interop.UIDropdownButton_Create()) { }

        ~UIDropdownButton() { Interop.UIDropdownButton_Destroy(mInstance); }

        public string Text
        {
            set { Interop.UIDropdownButton_SetText(mInstance, value); }
        }

        UIBaseImage mImage;
        public UIBaseImage Image
        {
            set { mImage = value; Interop.UIDropdownButton_SetImage(mInstance, mImage.Instance); }
        }

        public Math.vec4 TextColor
        {
            set { Interop.UIDropdownButton_SetTextColor(mInstance, value); }
        }

        UIComponent mContent;
        public UIComponent Content
        {
            set { mContent = value; Interop.UIDropdownButton_SetContent(mInstance, mContent.Instance); }
        }

        public Math.vec2 ContentSize
        {
            set { Interop.UIDropdownButton_SetContentSize(mInstance, value); }
        }
    }
}
