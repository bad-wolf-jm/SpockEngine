using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public class UIVec2Input : UIComponent
    {
        private bool mDerived = false;

        public UIVec2Input() : this(Interop.UIVec2Input_Create(), false) { }
        public UIVec2Input(IntPtr aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }

        ~UIVec2Input() { if (!mDerived) Interop.UIVec2Input_Destroy(mInstance); }

        public delegate void OnChangeDelegate();
        OnChangeDelegate onChange;
        public void OnChanged(OnChangeDelegate aHandler)
        {
            onChange = aHandler;

            Interop.UIVec2Input_OnChanged(mInstance, Marshal.GetFunctionPointerForDelegate(onChange));
        }

        public Math.vec2 Value
        {
            get { return Interop.UIVec2Input_GetValue(mInstance); }
            set { Interop.UIVec2Input_SetValue(mInstance, value); }
        }

        public Math.vec2 ResetValue
        {
            set { Interop.UIVec2Input_SetResetValues(mInstance, value); }
        }

        public string Format
        {
            set { Interop.UIVec2Input_SetFormat(mInstance, value); }
        }
    }

    public class UIVec3Input : UIComponent
    {
        private bool mDerived = false;

        public UIVec3Input() : this(Interop.UIVec3Input_Create(), false) { }
        public UIVec3Input(IntPtr aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }

        ~UIVec3Input() { if (!mDerived) Interop.UIVec3Input_Destroy(mInstance); }

        public delegate void OnChangeDelegate();
        OnChangeDelegate onChange;
        public void OnChanged(OnChangeDelegate aHandler)
        {
            onChange = aHandler;

            Interop.UIVec3Input_OnChanged(mInstance,  Marshal.GetFunctionPointerForDelegate(onChange));
        }

        public Math.vec2 Value
        {
            get { return Interop.UIVec3Input_GetValue(mInstance); }
            set { Interop.UIVec3Input_SetValue(mInstance, value); }
        }

        public Math.vec2 ResetValue
        {
            set { Interop.UIVec3Input_SetResetValues(mInstance, value); }
        }

        public string Format
        {
            set { Interop.UIVec3Input_SetFormat(mInstance, value); }
        }
    }

    public class UIVec4Input : UIComponent
    {
        private bool mDerived = false;

        public UIVec4Input() : this(Interop.UIVec4Input_Create(), false) { }
        public UIVec4Input(IntPtr aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }

        ~UIVec4Input() { if (!mDerived) Interop.UIVec4Input_Destroy(mInstance); }

        public delegate void OnChangeDelegate();
        OnChangeDelegate onChange;
        public void OnChanged(OnChangeDelegate aHandler)
        {
            onChange = aHandler;

            Interop.UIVec4Input_OnChanged(mInstance,  Marshal.GetFunctionPointerForDelegate(onChange));
        }

        public Math.vec2 Value
        {
            get { return Interop.UIVec4Input_GetValue(mInstance); }
            set { Interop.UIVec4Input_SetValue(mInstance, value); }
        }

        public Math.vec2 ResetValue
        {
            set { Interop.UIVec4Input_SetResetValues(mInstance, value); }
        }


        public string Format
        {
            set { Interop.UIVec4Input_SetFormat(mInstance, value); }
        }
    }
























}
