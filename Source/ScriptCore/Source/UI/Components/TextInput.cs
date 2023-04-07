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

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITextInput_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UITextInput_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextInput_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static string UITextInput_GetText(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextInput_OnTextChanged(ulong aInstance, OnChangeDelegate aHandler);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextInput_SetHintText(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextInput_SetTextColor(ulong aInstance, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UITextInput_SetBufferSize(ulong aInstance, uint aSize);
    }
}
