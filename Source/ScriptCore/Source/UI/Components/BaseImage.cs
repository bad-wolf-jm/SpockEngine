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

    }
}
