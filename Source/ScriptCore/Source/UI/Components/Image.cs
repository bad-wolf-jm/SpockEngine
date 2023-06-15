using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIImage : UIBaseImage
    {
        public UIImage() : base(Interop.UIImage_Create(), true) { }
        public UIImage(string aText, Math.vec2 aSize) : this()
        {
            SetImage(aText);
            Size = aSize;
        }

        ~UIImage() { Interop.UIImage_Destroy(mInstance); }
    }
}
