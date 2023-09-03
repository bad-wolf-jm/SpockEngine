using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIBaseImage : UIComponent
    {
        bool mDerived = false;
        public UIBaseImage() : base(Interop.UIBaseImage_Create()) { mDerived = false; }
        public UIBaseImage(IntPtr aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }
        public UIBaseImage(string aText, Math.vec2 aSize) : this() 
        { 
            SetImage(aText);
            Size = aSize;
        }

        ~UIBaseImage() { if (!mDerived) Interop.UIBaseImage_Destroy(mInstance); }

        public void SetImage(string aText) { Interop.UIBaseImage_SetImage(mInstance, aText); }

        public Math.vec2 Size
        {
            get { return Interop.UIBaseImage_GetSize(mInstance); }
            set { Interop.UIBaseImage_SetSize(mInstance, value); }
        }

        public Math.vec4 TintColor
        {
            get { return Interop.UIBaseImage_GetTintColor(mInstance); }
            set { Interop.UIBaseImage_SetTintColor(mInstance, value); }
        }

        public Math.vec2 TopLeft
        {
            get { return Interop.UIBaseImage_GetTopLeft(mInstance); }
            set { Interop.UIBaseImage_SetTopLeft(mInstance, value); }
        }

        public Math.vec2 BottomRight
        {
            get { return Interop.UIBaseImage_GetBottomRight(mInstance); }
            set { Interop.UIBaseImage_SetBottomRight(mInstance, value); }
        }

    }
}
