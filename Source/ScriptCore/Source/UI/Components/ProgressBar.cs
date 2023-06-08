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

        public void SetThickness(float aThickness) { UIProgressBar_SetThickness(mInstance, aThickness); }
    }
}
