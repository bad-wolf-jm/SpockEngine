using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIPropertyValue : UIBoxLayout
    {
        public UIPropertyValue() : base(Interop.UIPropertyValue_Create(), true) { }

        public UIPropertyValue(string aText) : base(Interop.UIPropertyValue_CreateWithText(aText), true) { }

        public UIPropertyValue(string aText, eBoxLayoutOrientation aOrientation) : base(Interop.UIPropertyValue_CreateWithTextAndOrientation(aText, aOrientation), true) { }

        ~UIPropertyValue() { Interop.UIPropertyValue_Destroy(mInstance); }

        public void SetValue(string aText) { Interop.UIPropertyValue_SetValue(mInstance, aText); }

        public void SetValueFont(eFontFamily aFont) { Interop.UIPropertyValue_SetValueFont(mInstance, aFont); }

        public void SetNameFont(eFontFamily aFont) { Interop.UIPropertyValue_SetNameFont(mInstance, aFont); }
    }
}
