using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIProgressBar : UIComponent
    {
        private bool mDerived = false;

        public UIProgressBar() : this(UIProgressBar_Create(), false) { }

        public UIProgressBar(ulong aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }

        ~UIProgressBar() { if (!mDerived) UIProgressBar_Destroy(mInstance); }

        public void SetText(string aText) { UIProgressBar_SetText(mInstance, aText); }

        public void SetTextColor(Math.vec4 aColor) { UIProgressBar_SetTextColor(mInstance, aColor); }

        public void SetProgressValue(float aValue) { UIProgressBar_SetProgressValue(mInstance, aValue); }

        public void SetProgressColor(Math.vec4 aColor) { UIProgressBar_SetTextColor(mInstance, aColor); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIProgressBar_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIProgressBar_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIProgressBar_SetText(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIProgressBar_SetTextColor(ulong aInstance, Math.vec4 aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIProgressBar_SetProgressValue(ulong aInstance, float aValue);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIProgressBar_SetProgressColor(ulong aInstance, Math.vec4 aColor);
    }
}
