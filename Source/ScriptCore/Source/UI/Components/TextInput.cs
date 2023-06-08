using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UITextInput : UIComponent
    {
        public UITextInput() : base(Interop.UITextInput_Create()) { }

        public UITextInput(string aText) : base(Interop.UITextInput_CreateWithText(aText)) { }

        ~UITextInput() { Interop.UITextInput_Destroy(mInstance); }

        public string Text
        {
            get { return Interop.UITextInput_GetText(mInstance); }
        }

        public void SetTextColor(Math.vec4 aColor)
        {
            Interop.UITextInput_SetTextColor(mInstance, aColor);
        }

        public void SetHintText(string aText)
        {
            Interop.UITextInput_SetHintText(mInstance, aText);
        }

        public void SetBufferSize(uint aSize)
        {
            Interop.UITextInput_SetBufferSize(mInstance, aSize);
        }

        public delegate bool OnChangeDelegate(string aValue);
        OnChangeDelegate onChanged;
        public void OnTextChanged(OnChangeDelegate aHandler)
        {
            onChanged = aHandler;

            Interop.UITextInput_OnTextChanged(mInstance, onChanged);
        }
    }
}
