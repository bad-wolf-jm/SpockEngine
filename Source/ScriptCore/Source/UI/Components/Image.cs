using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIImage : UIBaseImage
    {
        public UIImage() : base(UIImage_Create()) { }
        public UIImage(string aText, Math.vec2 aSize) : base(UIImage_CreateWithPath(aText, aSize)) { }

        ~UIImage() { UIImage_Destroy(mInstance); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIImage_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIImage_CreateWithPath(string aText, Math.vec2 Size);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImage_Destroy(ulong aInstance);
    }
}
