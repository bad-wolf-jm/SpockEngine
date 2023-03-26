using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIImageButton : UIBaseImage
    {
        public UIImageButton() : base(UIImageButton_Create()) { }
        public UIImageButton(string aText, Math.vec2 aSize) : base(UIImageButton_CreateWithPath(aText, aSize)) { }

        ~UIImageButton() { UIImageButton_Destroy(mInstance); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIImageButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIImageButton_CreateWithPath(string aText, Math.vec2 Size);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageButton_Destroy(ulong aInstance);
    }
}
