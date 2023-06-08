using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIProgressBar : UIComponent
    {
        private bool mDerived = false;

        public UIProgressBar() : this(Interop.UIProgressBar_Create(), false) { }

        public UIProgressBar(ulong aSelf, bool aDerived) : base(aSelf) { mDerived = aDerived; }

        ~UIProgressBar() { if (!mDerived) Interop.UIProgressBar_Destroy(mInstance); }

        public void SetText(string aText) { Interop.UIProgressBar_SetText(mInstance, aText); }

        public void SetTextColor(Math.vec4 aColor) { Interop.UIProgressBar_SetTextColor(mInstance, aColor); }

        public void SetProgressValue(float aValue) { Interop.UIProgressBar_SetProgressValue(mInstance, aValue); }

        public void SetProgressColor(Math.vec4 aColor) { Interop.UIProgressBar_SetTextColor(mInstance, aColor); }

        public void SetThickness(float aThickness) { Interop.UIProgressBar_SetThickness(mInstance, aThickness); }
    }
}
