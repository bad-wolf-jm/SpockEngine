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

        public Math.vec2 Size
        {
            get { return UIBaseImage_GetSize(mInstance); }
            set { UIBaseImage_SetSize(mInstance, value); }
        }

        public Math.vec4 TintColor
        {
            get { return UIBaseImage_GetTintColor(mInstance); }
            set { UIBaseImage_SetTintColor(mInstance, value); }
        }

        public Math.vec2 TopLeft
        {
            get { return UIBaseImage_GetTopLeft(mInstance); }
            set { UIBaseImage_SetTopLeft(mInstance, value); }
        }

        public Math.vec2 BottomRight
        {
            get { return UIBaseImage_GetBottomRight(mInstance); }
            set { UIBaseImage_SetBottomRight(mInstance, value); }
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIBaseImage_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIBaseImage_CreateWithPath(string aText, Math.vec2 Size);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_SetImage(ulong aInstance, string aPath);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_SetSize(ulong aInstance, Math.vec2 aSize);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static Math.vec2 UIBaseImage_GetSize(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_SetTopLeft(ulong aInstance, Math.vec2 aTopLeft);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static Math.vec2 UIBaseImage_GetTopLeft(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_SetBottomRight(ulong aInstance, Math.vec2 aBottomRight);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static Math.vec2 UIBaseImage_GetBottomRight(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBaseImage_SetTintColor(ulong aInstance, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static Math.vec4 UIBaseImage_GetTintColor(ulong aInstance);

    }
}
