using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIBaseImage : UIComponent
    {
        bool mDerived = false;
        public UIBaseImage() : base(UIBaseImage_Create()) { mDerived = false; }
        public UIBaseImage(ulong aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }
        public UIBaseImage(string aText, Math.vec2 aSize) : base(UIBaseImage_CreateWithPath(aText, aSize)) { }

        ~UIBaseImage() { if (!mDerived) UIBaseImage_Destroy(mInstance); }

        public void SetImage(string aText) { UIBaseImage_SetImage(mInstance, aText); }

        public void SetSize(float aWidth, float aHeight) { UIBaseImage_SetSize(mInstance, aWidth, aHeight); }

        public void SetTintColor(Math.vec4 aColor) { UIBaseImage_SetTintColor(mInstance, aColor); }

        public void SetRect(Math.vec2 aTopLeft, Math.vec2 aBottomRight) { UIBaseImage_SetRect(mInstance, aTopLeft, aBottomRight); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIBaseImage_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIBaseImage_CreateWithPath(string aText, Math.vec2 Size);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_SetImage(ulong aInstance, string aPath);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_SetSize(ulong aInstance, float aWidth, float aHeight);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_SetRect(ulong aInstance, Math.vec2 aTopLeft, Math.vec2 aBottomRight);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_SetTintColor(ulong aInstance, Math.vec4 aColor);
    }
}
