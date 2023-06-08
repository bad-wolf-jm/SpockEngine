using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UITextInput : UIComponent
    {
        public UITextInput() : base(UITextInput_Create()) { }

        public UITextInput(string aText) : base(UITextInput_CreateWithText(aText)) { }

        ~UITextInput() { UITextInput_Destroy(mInstance); }

        public string Text
        {
            get { return UITextInput_GetText(mInstance); }
        }

        public void SetTextColor(Math.vec4 aColor)
        {
            UITextInput_SetTextColor(mInstance, aColor);
        }

        public void SetHintText(string aText)
        {
            UITextInput_SetHintText(mInstance, aText);
        }

        public void SetBufferSize(uint aSize)
        {
            UITextInput_SetBufferSize(mInstance, aSize);
        }

        public delegate bool OnChangeDelegate(string aValue);
        OnChangeDelegate onChanged;
        public void OnTextChanged(OnChangeDelegate aHandler)
        {
            onChanged = aHandler;

            UITextInput_OnTextChanged(mInstance, onChanged);
        }
    }
}
