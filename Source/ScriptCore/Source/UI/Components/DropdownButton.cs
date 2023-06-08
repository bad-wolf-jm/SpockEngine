using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIDropdownButton : UIComponent
    {
        public UIDropdownButton() : base(UIDropdownButton_Create()) { }

        ~UIDropdownButton() { UIDropdownButton_Destroy(mInstance); }

        public string Text
        {
            set { UIDropdownButton_SetText(mInstance, value); }
        }

        UIBaseImage mImage;
        public UIBaseImage Image
        {
            set { mImage = value; UIDropdownButton_SetImage(mInstance, mImage.Instance); }
        }

        public Math.vec4 TextColor
        {
            set { UIDropdownButton_SetTextColor(mInstance, value); }
        }

        UIComponent mContent;
        public UIComponent Content
        {
            set { mContent = value; UIDropdownButton_SetContent(mInstance, mContent.Instance); }
        }

        public Math.vec2 ContentSize
        {
            set { UIDropdownButton_SetContentSize(mInstance, value); }
        }
    }
}
