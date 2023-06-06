using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIVec2Input : UIComponent
    {
        private bool mDerived = false;

        public UIVec2Input() : this(UIVec2Input_Create(), false) { }
        public UIVec2Input(ulong aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }

        ~UIVec2Input() { if (!mDerived) UIVec2Input_Destroy(mInstance); }

        public delegate void OnChangeDelegate();
        OnChangeDelegate onChange;
        public void OnChanged(OnChangeDelegate aHandler)
        {
            onChange = aHandler;

            UIVec2Input_OnChanged(mInstance, onChange);
        }

        public Math.vec2 Value
        {
            get { return UIVec2Input_GetValue(mInstance); }
            set { UIVec2Input_SetValue(mInstance, value); }
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVec2Input_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVec2Input_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec2Input_OnChanged(ulong aInstance, OnChangeDelegate aDelegate);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec2Input_SetValue(ulong aInstance, Math.vec2 aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static Math.vec2 UIVec2Input_GetValue(ulong aInstance);
    }

    public class UIVec3Input : UIComponent
    {
        private bool mDerived = false;

        public UIVec3Input() : this(UIVec3Input_Create(), false) { }
        public UIVec3Input(ulong aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }

        ~UIVec3Input() { if (!mDerived) UIVec3Input_Destroy(mInstance); }

        public delegate void OnChangeDelegate();
        OnChangeDelegate onChange;
        public void OnChanged(OnChangeDelegate aHandler)
        {
            onChange = aHandler;

            UIVec3Input_OnChanged(mInstance, onChange);
        }

        public Math.vec2 Value
        {
            get { return UIVec3Input_GetValue(mInstance); }
            set { UIVec3Input_SetValue(mInstance, value); }
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVec3Input_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVec3Input_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec3Input_OnChanged(ulong aInstance, OnChangeDelegate aDelegate);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec3Input_SetValue(ulong aInstance, Math.vec2 aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static Math.vec2 UIVec3Input_GetValue(ulong aInstance);
    }

    public class UIVec4Input : UIComponent
    {
        private bool mDerived = false;

        public UIVec4Input() : this(UIVec4Input_Create(), false) { }
        public UIVec4Input(ulong aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }

        ~UIVec4Input() { if (!mDerived) UIVec4Input_Destroy(mInstance); }

        public delegate void OnChangeDelegate();
        OnChangeDelegate onChange;
        public void OnChanged(OnChangeDelegate aHandler)
        {
            onChange = aHandler;

            UIVec4Input_OnChanged(mInstance, onChange);
        }

        public Math.vec2 Value
        {
            get { return UIVec4Input_GetValue(mInstance); }
            set { UIVec4Input_SetValue(mInstance, value); }
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVec4Input_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIVec4Input_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec4Input_OnChanged(ulong aInstance, OnChangeDelegate aDelegate);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIVec4Input_SetValue(ulong aInstance, Math.vec2 aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static Math.vec2 UIVec4Input_GetValue(ulong aInstance);
    }
























}
